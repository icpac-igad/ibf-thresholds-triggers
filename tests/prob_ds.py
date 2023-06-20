from utils import ens_mem_combiner
from utils import spi_prod_name_creator
from utils import ds_emprical_prbablity_creator
from utils import jjas_spi_prod_name_creator




output_path='probablity/'

spi_prod='mam'

lt_month=['nov','dec','jan','feb']

in_folder='data/'

threshold=[-0.03, -0.6,-0.9]

for ltm in lt_month:
    input_path=f'{in_folder}{spi_prod}/{ltm}_1981/'
    ds_ens=ens_mem_combiner(input_path)
    print(ds_ens)
    spi_prod_list=spi_prod_name_creator(ds_ens)
    print(spi_prod_list)
    ds_ens1 = ds_ens.assign_coords(spi_prod=('time',spi_prod_list))
    spi_ds_ens=ds_ens1.where(ds_ens1.spi_prod=='MAM', drop=True)
    ds_mild, ds_mod, ds_sev=ds_emprical_prbablity_creator(spi_ds_ens,5,threshold)
    ds_mild.to_netcdf(f'{output_path}{spi_prod}_{ltm}_mild.nc')
    ds_mod.to_netcdf(f'{output_path}{spi_prod}_{ltm}_mod.nc')
    ds_sev.to_netcdf(f'{output_path}{spi_prod}_{ltm}_sev.nc')
    ds_ens=[]
    spi_prod_list=[]
    ds_ens1=[]
    spi_ds_ens=[]
    ds_mild, ds_mod, ds_sev=[],[],[]


# output_path='/home/bulbul/Documents/work_data/uga_tt_project/kmj_seas5_1981_trail2/probablity/'

# spi_prod='jjas'

# lt_month=['mar','apr','may']

# in_folder='/home/bulbul/Documents/work_data/uga_tt_project/kmj_seas5_1981_trail2/'


# for ltm in lt_month:
#     input_path=f'{in_folder}{spi_prod}/{ltm}_1981/'
#     ds_ens=ens_mem_combiner(input_path)
#     print(ds_ens)
#     spi_prod_list=jjas_spi_prod_name_creator(ds_ens)
#     print(spi_prod_list)
#     ds_ens1 = ds_ens.assign_coords(spi_prod=('time',spi_prod_list))
#     spi_ds_ens=ds_ens1.where(ds_ens1.spi_prod=='JJAS', drop=True)
#     ds_mild, ds_mod, ds_sev=ds_emprical_prbablity_creator(spi_ds_ens,9)
#     ds_mild.to_netcdf(f'{output_path}{spi_prod}_{ltm}_mild.nc')
#     ds_mod.to_netcdf(f'{output_path}{spi_prod}_{ltm}_mod.nc')
#     ds_sev.to_netcdf(f'{output_path}{spi_prod}_{ltm}_sev.nc')
#     ds_ens=[]
#     spi_prod_list=[]
#     ds_ens1=[]
#     spi_ds_ens=[]
#     ds_mild, ds_mod, ds_sev=[],[],[]


# lt_month='nov'

# input_path=f'{in_folder}{spi_prod}/{lt_month}_1981/'

# ds_ens=ens_mem_combiner(input_path)

# spi_prod_list=spi_prod_name_creator(ds_ens)

# ds_ens1 = ds_ens.assign_coords(spi_prod=('time',spi_prod_list))

# spi_ds_ens=ds_ens1.where(ds_ens1.spi_prod=='MAM', drop=True)

# ds_mild, ds_mod, ds_sev=ds_emprical_prbablity_creator(spi_ds_ens)

# ds_mild.to_netcdf(f'{output_path}{spi_prod}_{lt_month}_mild.nc')

# ds_mod.to_netcdf(f'{output_path}{spi_prod}_{lt_month}_mod.nc')

# ds_sev.to_netcdf(f'{output_path}{spi_prod}_{lt_month}_sev.nc')
