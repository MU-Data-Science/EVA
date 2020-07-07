# Exhaustive Variant Analysis (EVA) on Human Genome Sequences

This site is ***under active development*** and will be updated in the coming months with code and documentation.

# Running variant analysis on human genomes using a single CloudLab node

1. Create an account on CloudLab by signing up [here](https://cloudlab.us/signup.php).  Select "Join Existing Project" with `EVA-public` as the project name.
2. By signing up, you agree to follow the [Acceptable Use Policy of CloudLab](https://cloudlab.us/aup.php).
3. After your account is approved, you can login to your account. Read the [CloudLab manual](http://docs.cloudlab.us/) on how to start an experiment.
4. Start an experiment using the profile `EVA-single-node-profile`.
You will need to select a machine type such as `xl170`, `c240g5`, etc. Also provide your CloudLab user name.
It will take a few minutes to start the experiment; so please be patient.

5. Open the shell for connecting to the CloudLab node in your browser.
6. Run the following commands on the shell:

    a. Clone the repo

       git clone https://github.com/MU-Data-Science/EVA.git

    b. Set up all the tools such as bwa, samtools, sambamba, Freebayes, etc.

       ~/EVA/scripts/setup_tools.sh

    c. Change directory to local block storage as we need ample space for running variant analysis.

       cd /mydata

    d. Set up and index the reference genome (e.g., hs38, hs38a, hs37). As this takes 30 minutes or so, use `screen` command to avoid disconnection.

       ~/EVA/scripts/setup_reference_genome.sh hs38

    f. Now download a whole genome sequence (paired-end) from The 1000 Genomes Project: `ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data`

       wget ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/HG00096/sequence_read/SRR062635_1.filt.fastq.gz
       wget ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/HG00096/sequence_read/SRR062635_2.filt.fastq.gz

    g. Run the variant analysis script by passing the required arguments.

       ~/EVA/scripts/run_variant_analysis.sh hs38 SRR062635

    h. The output of variant analysis is stored in a `.output.vcf` file. You can view this file using visualization tools such as [IGV](https://software.broadinstitute.org/software/igv/download).

# Running variant analysis on a cluster of CloudLab nodes

***💻 Under active development 🚧***

# Issues?

Please report them [here](https://github.com/MU-Data-Science/EVA/issues).

# Contributors

**Faculty PI:** Dr. Praveen Rao

**Ph.D. Students:** Arun Zachariah

# References
1. https://github.com/ekg/alignment-and-variant-calling-tutorial
2. https://github.com/biod/sambamba
3. https://github.com/lh3/bwa
4. https://github.com/ekg/freebayes
5. https://github.com/samtools/samtools
6. https://docs.brew.sh/Homebrew-on-Linux

# Acknowledgments

This work is supported by the National Science Foundation under [Grant No. 2034247](https://nsf.gov/awardsearch/showAward?AWD_ID=2034247).
