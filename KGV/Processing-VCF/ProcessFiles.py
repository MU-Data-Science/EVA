'''
Developed by Shivika Prasanna on 01/25/2022.
Last updated on 01/25/2022.
Reads VCF files and annotates using SNPEFF jar file. Stores output in the given argument.
Working code. 
Run in terminal as: python3 Code/ProcessFiles.py -i <input folder> -j <jar file>
input_folder = /path/to/Variant_Analysis_Output_Mar_8_2021_hg19/unzipped-VCF
jar_file = /path/to/snpEff/snpeff.jar

EX: python3 ProcessFiles.py -i /Users/shivikaprasanna/Desktop/Mizzou_Academics/GRA.nosync/Spring22-GRA/NSF-Rapid-Genomics/Variant_Analysis_Output_Mar_8_2021_hg19/sample/unzipped -o /Users/shivikaprasanna/Desktop/Mizzou_Academics/GRA.nosync/Spring22-GRA/NSF-Rapid-Genomics/Variant_Analysis_Output_Mar_8_2021_hg19/sample/annotated -j /Users/shivikaprasanna/Desktop/Mizzou_Academics/GRA.nosync/Spring22-GRA/NSF-Rapid-Genomics/snpEff/snpeff.jar
'''
import os, subprocess, shlex
from pathlib import Path
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="directory containing all VCF files")
ap.add_argument("-o", "--output", required=True, help="output directory path")
ap.add_argument("-j", "--jar", required=True, help="jar file")

args = vars(ap.parse_args())

input_unzipped_folder = str(args["input"])
output_annotated_folder = str(args["output"])
jar_file = str(args["jar"])

Path(output_annotated_folder).mkdir(parents=True, exist_ok=True)

for f in os.listdir(input_unzipped_folder): 
    input_unzipped_file = os.path.join(input_unzipped_folder, f)
    if f.endswith('.vcf'):
        output_vcf_file = f.replace('.vcf', '_ann_hg19.vcf')
        output_vcf_file_path = os.path.join(output_annotated_folder, output_vcf_file)
        print("JOIN:", output_vcf_file_path)
        with open (output_vcf_file_path, 'w+') as f:
            subprocess.call(shlex.split('java -Xmx8g -jar ' + jar_file + ' -v hg19 ' + input_unzipped_file), stdout=f)