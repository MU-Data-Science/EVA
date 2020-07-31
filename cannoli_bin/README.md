# Cannoli
Cannoli enables distributed execution of bioinformatics tools, such as bwa and freebayes on Apache Spark. This directory holds modifed version of the original `cannoli-shell` and `cannoli-submit` executables. More details about cannoli could be found at http://bdgenomics.org 

## To update the jars
1. Clone the Cannoli fork.
    ```
    git clone https://github.com/Arun-George-Zachariah/cannoli.git)
    ```
2. Build and install the code:
    ```
    bash build.sh
    ```
3. Replace `cannoli-assembly-spark3_2.12-0.11.0-SNAPSHOT.jar` with `cannoli/assembly/cannoli-assembly-*-SNAPSHOT.jar`