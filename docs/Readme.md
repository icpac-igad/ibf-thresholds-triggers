## The tex template to convert pdf into docx
1. The template is based on https://github.com/AndyClifton/CorporateLaTeX
1. tex is handy to use for docs having many number of images to attach in the document
2. Code the images in text, convert into pdf using texstudio, then convert
   into html and then docx using pdftohtml and pandoc
   
   ```
   pdftohtml -enc UTF-8 -noframes report.pdf report.html
   pandoc  -s report.html -o report.docx
   ```
3. Few corrections needed to make in the docx before sending for review 
