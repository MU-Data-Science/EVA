# Processing VCF and CADD files

Once the RNA-sequencing has been completed, the VCF and CADD files are further processed. [SnpEff](https://pcingola.github.io/SnpEff/se_introduction/) is a variant annotation and effect prediction tool that annotates and predicts the effects of genetic variants. We run the VCF files first through the tool and then further process it using [vcf2rdf](https://www.roelj.com/sparqling-genomics.html) to generate triples. vcf2rdf is a tool available in [SPARQLing-genomics](https://github.com/UMCUGenetics/sparqling-genomics), a combination of multiple tools to create an integrated knowledge graph (KG). For CADD scores, we have defined an ontology that helps us in writing triples to add to our KG. 

It is advised to use Cloudlab for processing these files due to their size.

## Process VCF files

We have a script that first unzips all the VCF files. Then, the files are processing using the snpEff jar file. Next, vcf2rdf tool is called to annotate on the files. Once the annotation is completed, triples for each sequence are stored in an N3 format. These N3 files are then converted to NQ format, to easily identify triples with a named graph. 

Add the snpEff.jar path at Line 20 and vcf2rdf path at Line 27.

To run the script:

    $ ./process_vcfs.sh /path/to/VCF/folder

Since it is a lengthy process, it is recommended to use either Screen commands or as a background task using '&'.

## Process CADD scores

CADD scores are computed separately when processing the sequences and hence, are stored in separate files. To process these files to create triples in Turtle formant, run as:

    $ ./process_cadd.sh /path/to/CADD/folder 


NOTE: Please ensure all the zipped files are under a folder.

# Extracting GENE_ID as a Triple from the VCF files

To generate the triples, run as following:

    $  python3 Code/GeneIDasTriple.py -i /input/VCF/directory -o /output/file/path.nq