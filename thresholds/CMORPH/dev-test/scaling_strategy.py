#!/usr/bin/env python3
"""
Multi-Year Scaling Strategy for NOAA CDR Precipitation Processing
Extends the precipitation processor for efficient large-scale processing
"""

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import concurrent.futures
import time
from dataclasses import dataclass
import multiprocessing as mp

from precipitation_processor_architecture import PrecipitationProcessor, CMORPHProcessor, PERSIANNProcessor

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingTask:
    """Individual processing task for a dataset and year"""
    dataset_name: str
    year: int
    priority: int = 1  # Higher numbers = higher priority
    estimated_files: int = 0
    estimated_time_hours: float = 0.0
    dependencies: List[str] = None  # List of task IDs this depends on
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
    
    @property
    def task_id(self) -> str:
        return f"{self.dataset_name}_{self.year}"


@dataclass
class ProcessingResult:
    """Result of a processing task"""
    task_id: str
    dataset_name: str
    year: int
    success: bool
    start_time: datetime
    end_time: datetime
    files_processed: int
    files_successful: int
    files_failed: int
    error_message: Optional[str] = None
    
    @property
    def duration_minutes(self) -> float:
        return (self.end_time - self.start_time).total_seconds() / 60.0
    
    @property
    def success_rate(self) -> float:
        if self.files_processed == 0:
            return 0.0
        return (self.files_successful / self.files_processed) * 100.0


class ScalingProcessor:
    """Manages large-scale multi-year processing with parallelization and optimization"""
    
    def __init__(self, output_dir: str = ".", max_workers: int = None, 
                 batch_size: int = 50, enable_checkpointing: bool = True):
        """
        Initialize scaling processor
        
        Args:
            output_dir: Base output directory
            max_workers: Maximum number of parallel workers
            batch_size: Number of files to process per batch
            enable_checkpointing: Enable checkpoint/resume functionality
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Determine optimal number of workers
        if max_workers is None:
            max_workers = max(1, min(mp.cpu_count() - 1, 4))  # Leave one CPU free, max 4 workers
        self.max_workers = max_workers
        
        self.batch_size = batch_size
        self.enable_checkpointing = enable_checkpointing
        
        # Checkpoint file
        self.checkpoint_file = self.output_dir / "processing_checkpoint.json"
        self.completed_tasks = self.load_checkpoint()
        
        logger.info(f"Scaling processor initialized with {max_workers} workers")
        logger.info(f"Batch size: {batch_size}, Checkpointing: {enable_checkpointing}")
    
    def load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint data from previous run"""
        if not self.enable_checkpointing or not self.checkpoint_file.exists():
            return {}
        
        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            logger.info(f"Loaded checkpoint with {len(checkpoint)} completed tasks")
            return checkpoint
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return {}
    
    def save_checkpoint(self, task_result: ProcessingResult):
        """Save checkpoint after completing a task"""
        if not self.enable_checkpointing:
            return
        
        self.completed_tasks[task_result.task_id] = {
            'dataset_name': task_result.dataset_name,
            'year': task_result.year,
            'success': task_result.success,
            'completion_time': task_result.end_time.isoformat(),
            'files_processed': task_result.files_processed,
            'files_successful': task_result.files_successful,
            'duration_minutes': task_result.duration_minutes
        }
        
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(self.completed_tasks, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def estimate_dataset_files(self, dataset_name: str, year: int) -> int:
        """
        Estimate number of files for a dataset/year combination
        
        Args:
            dataset_name: Dataset name ('cmorph' or 'persiann')
            year: Target year
            
        Returns:
            Estimated number of NetCDF files
        """
        # Rough estimates based on dataset characteristics
        estimates = {
            'cmorph': {
                'files_per_day': 48,  # 30-minute intervals = 48 files/day
                'availability_start': 1998
            },
            'persiann': {
                'files_per_day': 1,   # Daily files
                'availability_start': 1983
            }
        }
        
        if dataset_name not in estimates:
            return 365  # Default estimate
        
        config = estimates[dataset_name]
        
        # Check if year is in available range
        if year < config['availability_start']:
            return 0  # No data available
        
        # Calculate days in year
        is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        days_in_year = 366 if is_leap else 365
        
        return days_in_year * config['files_per_day']
    
    def estimate_processing_time(self, dataset_name: str, estimated_files: int) -> float:
        """
        Estimate processing time in hours
        
        Args:
            dataset_name: Dataset name
            estimated_files: Number of files to process
            
        Returns:
            Estimated processing time in hours
        """
        # Base processing time per file (empirical estimates)
        time_per_file_minutes = {
            'cmorph': 0.5,    # 30 seconds per file (high frequency data)
            'persiann': 1.0   # 1 minute per file (daily aggregated data)
        }
        
        base_time = time_per_file_minutes.get(dataset_name, 0.75)
        
        # Factor in parallelization efficiency (diminishing returns)
        parallel_efficiency = 0.8 if self.max_workers > 1 else 1.0
        
        total_minutes = (estimated_files * base_time) / (self.max_workers * parallel_efficiency)
        
        return total_minutes / 60.0  # Convert to hours
    
    def create_processing_plan(self, datasets: List[str], year_ranges: Dict[str, Tuple[int, int]]) -> List[ProcessingTask]:
        """
        Create comprehensive processing plan with task prioritization
        
        Args:
            datasets: List of dataset names to process
            year_ranges: Dictionary mapping dataset names to (start_year, end_year) tuples
            
        Returns:
            List of processing tasks sorted by priority
        """
        tasks = []
        
        for dataset_name in datasets:
            if dataset_name not in year_ranges:
                logger.warning(f"No year range specified for {dataset_name}, skipping")
                continue
            
            start_year, end_year = year_ranges[dataset_name]
            
            for year in range(start_year, end_year + 1):
                # Skip if already completed
                task_id = f"{dataset_name}_{year}"
                if task_id in self.completed_tasks:
                    logger.info(f"Task {task_id} already completed, skipping")
                    continue
                
                estimated_files = self.estimate_dataset_files(dataset_name, year)
                if estimated_files == 0:
                    logger.warning(f"No files estimated for {dataset_name} {year}, skipping")
                    continue
                
                estimated_time = self.estimate_processing_time(dataset_name, estimated_files)
                
                # Priority based on year (more recent = higher priority) and dataset size
                base_priority = year - 1980  # Baseline priority based on year
                if dataset_name == 'cmorph':
                    base_priority += 10  # Higher resolution data gets higher priority
                
                task = ProcessingTask(
                    dataset_name=dataset_name,
                    year=year,
                    priority=base_priority,
                    estimated_files=estimated_files,
                    estimated_time_hours=estimated_time
                )
                
                tasks.append(task)
        
        # Sort by priority (highest first)
        tasks.sort(key=lambda x: x.priority, reverse=True)
        
        logger.info(f"Created processing plan with {len(tasks)} tasks")
        total_estimated_time = sum(task.estimated_time_hours for task in tasks)
        logger.info(f"Total estimated processing time: {total_estimated_time:.1f} hours ({total_estimated_time/24:.1f} days)")
        
        return tasks
    
    def process_single_task(self, task: ProcessingTask) -> ProcessingResult:
        """
        Process a single task (dataset + year combination)
        
        Args:
            task: Processing task to execute
            
        Returns:
            Processing result
        """
        start_time = datetime.now()
        logger.info(f"Starting task {task.task_id} (Priority: {task.priority})")
        
        try:
            # Initialize appropriate processor
            if task.dataset_name == 'cmorph':
                processor = CMORPHProcessor(
                    target_year=task.year,
                    output_dir=str(self.output_dir),
                    use_s3_direct=True
                )
            elif task.dataset_name == 'persiann':
                processor = PERSIANNProcessor(
                    target_year=task.year,
                    output_dir=str(self.output_dir),
                    use_s3_direct=True
                )
            else:
                raise ValueError(f"Unknown dataset: {task.dataset_name}")
            
            # Get NetCDF files
            nc_files = processor.get_nc_files()
            if not nc_files:
                raise Exception(f"No NetCDF files found for {task.dataset_name} {task.year}")
            
            # Process files in batches to manage memory
            total_files = len(nc_files)
            successful_files = 0
            failed_files = 0
            
            for batch_start in range(0, total_files, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total_files)
                batch_files = nc_files[batch_start:batch_end]
                
                logger.info(f"Processing batch {batch_start//self.batch_size + 1}: files {batch_start+1}-{batch_end} of {total_files}")
                
                # Process batch
                summary = processor.process_individual_files(batch_files, max_files=len(batch_files))
                
                successful_files += summary['processing_summary']['successful_files']
                failed_files += summary['processing_summary']['failed_files']
                
                # Small delay between batches to prevent overwhelming the system
                if batch_end < total_files:
                    time.sleep(1)
            
            end_time = datetime.now()
            
            result = ProcessingResult(
                task_id=task.task_id,
                dataset_name=task.dataset_name,
                year=task.year,
                success=successful_files > 0,
                start_time=start_time,
                end_time=end_time,
                files_processed=total_files,
                files_successful=successful_files,
                files_failed=failed_files
            )
            
            logger.info(f"Completed task {task.task_id}: {successful_files}/{total_files} files successful ({result.success_rate:.1f}%)")
            
            return result
            
        except Exception as e:
            end_time = datetime.now()
            
            result = ProcessingResult(
                task_id=task.task_id,
                dataset_name=task.dataset_name,
                year=task.year,
                success=False,
                start_time=start_time,
                end_time=end_time,
                files_processed=0,
                files_successful=0,
                files_failed=0,
                error_message=str(e)
            )
            
            logger.error(f"Task {task.task_id} failed: {e}")
            return result
    
    def process_tasks_parallel(self, tasks: List[ProcessingTask]) -> List[ProcessingResult]:
        """
        Process multiple tasks in parallel
        
        Args:
            tasks: List of processing tasks
            
        Returns:
            List of processing results
        """
        logger.info(f"Starting parallel processing of {len(tasks)} tasks with {self.max_workers} workers")
        
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(self.process_single_task, task): task for task in tasks}
            
            # Process completed tasks
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Save checkpoint
                    self.save_checkpoint(result)
                    
                    # Log progress
                    completed_count = len(results)
                    remaining_count = len(tasks) - completed_count
                    logger.info(f"Progress: {completed_count}/{len(tasks)} tasks completed, {remaining_count} remaining")
                    
                except Exception as e:
                    logger.error(f"Task {task.task_id} generated an exception: {e}")
                    # Create error result
                    error_result = ProcessingResult(
                        task_id=task.task_id,
                        dataset_name=task.dataset_name,
                        year=task.year,
                        success=False,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        files_processed=0,
                        files_successful=0,
                        files_failed=0,
                        error_message=str(e)
                    )
                    results.append(error_result)
        
        return results
    
    def generate_scaling_report(self, results: List[ProcessingResult]) -> Dict[str, Any]:
        """
        Generate comprehensive scaling report
        
        Args:
            results: List of processing results
            
        Returns:
            Scaling report dictionary
        """
        total_tasks = len(results)
        successful_tasks = sum(1 for r in results if r.success)
        failed_tasks = total_tasks - successful_tasks
        
        total_files_processed = sum(r.files_processed for r in results)
        total_files_successful = sum(r.files_successful for r in results)
        total_processing_time = sum(r.duration_minutes for r in results)
        
        # Dataset breakdown
        dataset_stats = {}
        for result in results:
            dataset = result.dataset_name
            if dataset not in dataset_stats:
                dataset_stats[dataset] = {
                    'tasks': 0,
                    'successful_tasks': 0,
                    'files_processed': 0,
                    'files_successful': 0,
                    'total_time_minutes': 0.0
                }
            
            stats = dataset_stats[dataset]
            stats['tasks'] += 1
            if result.success:
                stats['successful_tasks'] += 1
            stats['files_processed'] += result.files_processed
            stats['files_successful'] += result.files_successful
            stats['total_time_minutes'] += result.duration_minutes
        
        # Performance metrics
        avg_files_per_minute = total_files_successful / total_processing_time if total_processing_time > 0 else 0
        
        report = {
            'scaling_report': {
                'generation_timestamp': datetime.now().isoformat(),
                'processing_summary': {
                    'total_tasks': total_tasks,
                    'successful_tasks': successful_tasks,
                    'failed_tasks': failed_tasks,
                    'task_success_rate': f"{successful_tasks/total_tasks*100:.1f}%" if total_tasks > 0 else "0%"
                },
                'file_processing_summary': {
                    'total_files_processed': total_files_processed,
                    'total_files_successful': total_files_successful,
                    'total_files_failed': total_files_processed - total_files_successful,
                    'file_success_rate': f"{total_files_successful/total_files_processed*100:.1f}%" if total_files_processed > 0 else "0%"
                },
                'performance_metrics': {
                    'total_processing_time_hours': total_processing_time / 60.0,
                    'average_files_per_minute': avg_files_per_minute,
                    'average_task_duration_minutes': total_processing_time / total_tasks if total_tasks > 0 else 0,
                    'parallel_workers_used': self.max_workers
                },
                'dataset_breakdown': dataset_stats,
                'scaling_efficiency': {
                    'theoretical_sequential_time_hours': total_processing_time / 60.0,
                    'actual_parallel_time_hours': max([r.duration_minutes for r in results]) / 60.0 if results else 0,
                    'parallelization_speedup': f"{total_processing_time / max([r.duration_minutes for r in results]):.1f}x" if results and max([r.duration_minutes for r in results]) > 0 else "N/A"
                }
            },
            'detailed_results': [
                {
                    'task_id': r.task_id,
                    'dataset': r.dataset_name,
                    'year': r.year,
                    'success': r.success,
                    'files_processed': r.files_processed,
                    'files_successful': r.files_successful,
                    'success_rate': f"{r.success_rate:.1f}%",
                    'duration_minutes': r.duration_minutes,
                    'error_message': r.error_message
                } for r in results
            ]
        }
        
        return report


def main():
    """CLI for scaling processor"""
    parser = argparse.ArgumentParser(
        description="Multi-year scaling processor for NOAA CDR precipitation datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process CMORPH 1983-1985
  python scaling_strategy.py --datasets cmorph --year-ranges cmorph:1983-1985
  
  # Process both datasets with different ranges
  python scaling_strategy.py --datasets cmorph persiann --year-ranges cmorph:1998-2000 persiann:1983-1985
  
  # Full parallel processing with 8 workers
  python scaling_strategy.py --datasets cmorph --year-ranges cmorph:1998-2010 --max-workers 8
  
  # Resume processing from checkpoint
  python scaling_strategy.py --datasets cmorph --year-ranges cmorph:1998-2010 --resume
        """
    )
    
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['cmorph', 'persiann'],
        required=True,
        help='Datasets to process'
    )
    
    parser.add_argument(
        '--year-ranges',
        nargs='+',
        required=True,
        help='Year ranges for each dataset (format: dataset:start-end, e.g., cmorph:1998-2000)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='.',
        help='Base output directory for all processed data'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        help='Maximum number of parallel workers (default: auto-detect)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Number of files to process per batch (default: 50)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint (skip completed tasks)'
    )
    
    parser.add_argument(
        '--disable-checkpointing',
        action='store_true',
        help='Disable checkpoint/resume functionality'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse year ranges
    year_ranges = {}
    for range_spec in args.year_ranges:
        try:
            dataset, years = range_spec.split(':')
            start_year, end_year = map(int, years.split('-'))
            year_ranges[dataset] = (start_year, end_year)
        except ValueError:
            logger.error(f"Invalid year range format: {range_spec}. Use format dataset:start-end")
            exit(1)
    
    logger.info("Multi-Year Scaling Processor Starting")
    logger.info(f"Datasets: {', '.join(args.datasets)}")
    logger.info(f"Year ranges: {year_ranges}")
    
    # Initialize scaling processor
    scaling_processor = ScalingProcessor(
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        enable_checkpointing=not args.disable_checkpointing
    )
    
    # Create processing plan
    tasks = scaling_processor.create_processing_plan(args.datasets, year_ranges)
    
    if not tasks:
        logger.warning("No tasks to process")
        exit(0)
    
    logger.info(f"Starting processing of {len(tasks)} tasks")
    
    # Process tasks
    start_time = datetime.now()
    results = scaling_processor.process_tasks_parallel(tasks)
    end_time = datetime.now()
    
    # Generate report
    report = scaling_processor.generate_scaling_report(results)
    
    # Save report
    report_filename = f"scaling_report_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
    report_path = Path(args.output_dir) / report_filename
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    summary = report['scaling_report']['processing_summary']
    performance = report['scaling_report']['performance_metrics']
    
    logger.info(f"\\n{'='*60}")
    logger.info(f"SCALING PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total runtime: {(end_time - start_time).total_seconds()/3600:.1f} hours")
    logger.info(f"Tasks completed: {summary['successful_tasks']}/{summary['total_tasks']} ({summary['task_success_rate']})")
    logger.info(f"Files processed: {report['scaling_report']['file_processing_summary']['total_files_successful']}")
    logger.info(f"Processing rate: {performance['average_files_per_minute']:.1f} files/minute")
    logger.info(f"Detailed report saved: {report_path}")
    
    # Exit code based on success rate
    task_success_rate = float(summary['task_success_rate'].rstrip('%'))
    if task_success_rate >= 90:
        exit(0)
    elif task_success_rate >= 70:
        exit(1)
    else:
        exit(2)


if __name__ == "__main__":
    main()