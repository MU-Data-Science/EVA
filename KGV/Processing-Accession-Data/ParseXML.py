'''
Developed by Shivika Prasanna on 02/25/2022.
Last updated on 03/21/2022.
Working code.
Parse all .xml files and populate KG using the ontology in definition.n3 file.
Run as: python3 ParseXML.py -i <input path to directory> -o <output path for n3 file>
'''

import os
import pandas as pd

import argparse

from lxml import etree
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import FOAF , XSD

from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="input path to directory")
ap.add_argument("-o", "--output", required=True, help="output path for n3 file")

args = vars(ap.parse_args())
dir_name = str(args['input'])
output_path = str(args['output'])

g = Graph()
b = Namespace('http://sg.org/')

for filename in os.listdir(dir_name):
    if filename.endswith('.xml'):
        path = os.path.join(dir_name, filename)
        print("Currently processing: ", path)
        doc = etree.parse(path)

        if (doc.find('STUDY')):
            study_elem = doc.find('STUDY')
            
            uri = URIRef('http://sg.org/s{}'.format(study_elem.get('accession')))
            g.add((uri, RDF.type, b.study))
                    
            for item in study_elem.find('IDENTIFIERS'):
                if 'PRIMARY_ID' in item.tag:
                    g.add((uri, b.has_primary_id, Literal(item.text)))
                elif 'SECONDARY_ID' in item.tag:
                    g.add((uri, b.has_secondary_id, Literal(item.text)))
                elif 'SUBMITTER_ID' in item.tag:
                    g.add((uri, b.has_submitter_id, Literal(item.text)))
                elif 'EXTERNAL_ID' in item.tag:
                    g.add((uri, b.has_external_id, Literal(item.text)))
                else:
                    print("Missing in IDENTIFIERS: ", item.tag) 

            for item in study_elem.find('DESCRIPTOR'):
                if 'STUDY_TITLE' in item.tag:
                    g.add((uri, b.has_title, Literal(item.text)))
                elif 'STUDY_ABSTRACT' in item.tag:
                    g.add((uri, b.has_abstract, Literal(item.text)))
                elif 'CENTER_PROJECT_NAME' in item.tag:
                    g.add((uri, b.has_center_project_name, Literal(item.text)))
                elif 'STUDY_DESCRIPTION' in item.tag:
                    g.add((uri, b.has_description, Literal(item.text)))
                else:
                    print("Missing in DESCRIPTOR: ", item.tag) 

            count = 0
            for items in study_elem.find('STUDY_LINKS'):
                for item in items:
                    if 'URL_LINK' in item.tag:
                        url_ref = URIRef(uri+"/url_links{}".format(count))
                        g.add((url_ref, RDF.type, b.url_link))
                        g.add((uri, b.has_study_links, url_ref))
                        for i in item:
                            if 'LABEL' in i.tag:
                                g.add((url_ref, b.has_label, Literal(i.text)))
                            elif 'URL' in i.tag:
                                g.add((url_ref, b.has_url, Literal(i.text)))
                            else:
                                print("Missing in URL_LINK: ", item.tag) 
                    elif 'XREF_LINK' in item.tag:
                        xref_ref = URIRef(uri+"/xref_links{}".format(count))
                        g.add((xref_ref, RDF.type, b.xref_link))
                        g.add((uri, b.has_study_links, xref_ref))
                        for i in item:
                            if 'DB' in i.tag:
                                g.add((xref_ref, b.has_db, Literal(i.text)))
                            elif 'ID' in i.tag:
                                g.add((xref_ref, b.has_xref_link_id, Literal(i.text)))
                            else:
                                print("Missing in XREF_LINK: ", item.tag) 
                    else:
                        print("Missing in STUDY_LINKS: ", item.tag) 
                count += 1

            count = 0
            for items in study_elem.find('STUDY_ATTRIBUTES'):
                for item in items:
                    study_ref = URIRef(uri+"/study_attribute{}".format(count))
                    g.add((study_ref, RDF.type, b.study_attribute))
                    g.add((uri, b.has_study_attribute, study_ref))
                    if 'TAG' in item.tag:
                        g.add((study_ref, b.has_tag, Literal(item.text)))
                    elif 'VALUE' in item.tag:
                        g.add((study_ref, b.has_value, Literal(item.text)))
                    else:
                        print("Missing in STUDY_ATTRIBUTES: ", item.tag) 
                count += 1

        if (doc.find('EXPERIMENT')):
            exp_elem = doc.find('EXPERIMENT')
            uri = URIRef('http://sg.org/s{}'.format(exp_elem.get('accession')))
            g.add((uri, RDF.type, b.experiment))

            for item in exp_elem.find('IDENTIFIERS'):
                if 'PRIMARY_ID' in item.tag:
                    g.add((uri, b.has_primary_id, Literal(item.text)))
                elif 'SUBMITTER_ID' in item.tag:
                    g.add((uri, b.has_submitter_id, Literal(item.text)))
                else:
                    print("Missing in IDENTIFIERS: ", item.tag)

            if (len(exp_elem.find('TITLE'))) is not None:
                g.add((uri, b.has_title, Literal(exp_elem.find('TITLE').text)))   

            for items in exp_elem.find('STUDY_REF'):
                g.add((uri, b.exp_has_study_ref, b.study_ref))
                for item in items:
                    if 'PRIMARY_ID' in item.tag:
                        g.add((uri, b.study_ref_primary_id, Literal(item.text)))
                    elif 'SECONDARY_ID' in item.tag:
                        g.add((uri, b.study_ref_secondary_id, Literal(item.text)))
                    else:
                         print("Missing in IDENTIFIERS: ", item.tag)

            count = 0
            for item in exp_elem.find('DESIGN'):
                if 'DESIGN_DESCRIPTION' in item.tag:
                    g.add((uri, b.has_design_description, Literal(item.text)))   
                elif 'SAMPLE_DESCRIPTOR' in item.tag:
                    # g.add((uri, b.design_has_descriptors, b.sample_descriptor))
                    for i in item:
                        if 'IDENTIFIERS' in i.tag:
                            for tags in i:
                                if 'PRIMARY_ID' in tags.tag:
                                    g.add((uri, b.has_sample_descriptor_primary_id, Literal(tags.text)))
                                elif 'EXTERNAL_ID' in tags.tag:
                                    g.add((uri, b.has_sample_descriptor_external_id, Literal(tags.text)))
                        else:
                            print("Missing in SAMPLE_DESCRIPTOR: ", i.tag)
                elif 'LIBRARY_DESCRIPTOR' in item.tag:
                    # g.add((uri, b.design_has_descriptors, b.library_descriptor))
                    for i in item:
                        if 'LIBRARY_NAME' in i.tag:
                            g.add((uri, b.has_library_name, Literal(i.text)))
                        elif 'LIBRARY_STRATEGY' in i.tag:
                            g.add((uri, b.has_library_strategy, Literal(i.text)))
                        elif 'LIBRARY_SOURCE' in i.tag:
                            g.add((uri, b.has_library_source, Literal(i.text)))
                        elif 'LIBRARY_SELECTION' in i.tag:
                            g.add((uri, b.has_library_selection, Literal(i.text)))
                        elif 'LIBRARY_LAYOUT' in i.tag:
                            g.add((uri, b.has_library_layout, Literal(i.text)))
                        elif 'LIBRARY_CONSTRUCTION_PROTOCOL' in i.tag:
                            g.add((uri, b.has_library_construction_protocol, Literal(i.text)))
                        else:
                            print("Missing in LIBRARY_DESCRIPTOR: ", i.tag)
                         
                for item in exp_elem.find('PLATFORM'):
                    if 'ILLUMINA' in item.tag:
                        for i in item:
                            if 'INSTRUMENT_MODEL' in i:
                                g.add((uri, b.has_instrument_model, Literal(i.text)))
                            else:
                                print("Missing in INSTRUMENT_MODEL: ", i.tag)
                    else:
                        print("Missing in ILLUMINA: ", item.tag)

                count = 0
                for items in exp_elem.find('EXPERIMENT_LINKS'):
                    for item in items:
                        if 'XREF_LINK' in item.tag:
                            xref_ref = URIRef(uri+"/xref_links{}".format(count))
                            g.add((xref_ref, RDF.type, b.xref_link))
                            g.add((uri, b.has_experiment_link, xref_ref))
                            for i in item:
                                if 'DB' in i.tag:
                                    g.add((xref_ref, b.has_db, Literal(i.text)))
                                elif 'ID' in i.tag:
                                    g.add((xref_ref, b.has_xref_link_id, Literal(i.text)))
                                else:
                                    print("Missing in XREF_LINK: ", i.tag) 
                        else:
                            print("Missing in EXPERIMENT_LINKS: ", item.tag) 
                    count += 1

                count = 0
                for item in exp_elem.find('EXPERIMENT_ATTRIBUTES'):
                    if 'EXPERIMENT_ATTRIBUTE' in item.tag:
                        exp_ref = URIRef(uri+"/exp_attributes{}".format(count))
                        g.add((exp_ref, RDF.type, b.experiment_attribute))
                        g.add((uri, b.has_experiment_attribute, exp_ref))
                        for i in item:
                            if 'TAG' in i.tag:
                                g.add((exp_ref, b.has_tag, Literal(i.text)))
                            elif 'VALUE' in i.tag:
                                g.add((exp_ref, b.has_value, Literal(i.text)))
                            else:
                                print("Missing in : EXPERIMENT_ATTRIBUTE", i.tag) 
                    else:
                        print("Missing in EXPERIMENT_ATTRIBUTES: ", item.tag) 
                    count += 1
                

        if (doc.find('RUN')):
            run_elem = doc.find('RUN')
            uri = URIRef('http://sg.org/s{}'.format(run_elem.get('accession')))
            g.add((uri, RDF.type, b.experiment))

            if(run_elem.find('IDENTIFIERS')):
                for item in run_elem.find('IDENTIFIERS'):
                    if 'PRIMARY_ID' in item.tag:
                        g.add((uri, b.has_primary_id, Literal(item.text)))
                    elif 'SUBMITTER_ID' in item.tag:
                        g.add((uri, b.has_submitter_id, Literal(item.text)))
                    else:
                        print("Missing in IDENTIFIERS: ", item.tag)

            if (len(run_elem.find('TITLE'))) is not None:
                g.add((uri, b.has_title, Literal(run_elem.find('TITLE').text)))   
            
            if(run_elem.find('EXPERIMENT_REF')):
                for items in run_elem.find('EXPERIMENT_REF'):
                    g.add((uri, b.run_has_exp_ref, b.experiment_ref))
                    for item in items:
                        if 'PRIMARY_ID' in item.tag:
                            g.add((uri, b.exp_ref_primary_id, Literal(item.text)))
                        else:
                            print("Missing in IDENTIFIERS: ", item.tag)
            
            if (run_elem.find('PLATFORM')):
                for item in run_elem.find('PLATFORM'):
                    if 'ILLUMINA' in item.tag:
                        if 'INSTRUMENT_MODEL' in item.tag:
                            g.add((uri, b.has_instrument_model, Literal(i.text)))
                        else:
                            print("Missing in INSTRUMENT_MODEL: ", item.tag)
                    else:
                        print("Missing in ILLUMINA: ", item.tag)

            if(run_elem.find('DATA_BLOCK')): 
                count = 0
                for items in run_elem.find('DATA_BLOCK'):
                    for item in items:
                        if 'FILE' in item.tag:
                            file_count = URIRef(uri+"/data_block_file{}".format(count))
                            g.add((file_count, RDF.type, b.data_block_files))
                            g.add((uri, b.has_file, file_count))
                            g.add((uri, b.has_file, Literal(item.text)))
                        else:
                            print("Missing in DATA_BLOCK: ", item.tag)
                    count += 1
            
            if(run_elem.find('RUN_LINKS')):
                count = 0
                for items in run_elem.find('RUN_LINKS'):
                    for item in items:
                        if 'XREF_LINK' in item.tag:
                            xref_ref = URIRef(uri+"/xref_links{}".format(count))
                            g.add((xref_ref, RDF.type, b.xref_link))
                            g.add((uri, b.has_xref_link, xref_ref))
                            for i in item:
                                if 'DB' in i.tag:
                                    g.add((xref_ref, b.has_db, Literal(i.text)))
                                elif 'ID' in i.tag:
                                    g.add((xref_ref, b.has_xref_link_id, Literal(i.text)))
                                else:
                                    print("Missing in XREF_LINK: ", i.tag) 
                        else:
                            print("Missing in RUN_LINKS: ", item.tag) 
                    count += 1
            
            if(run_elem.find('RUN_ATTRIBUTES')):
                count = 0
                for items in run_elem.find('RUN_ATTRIBUTES'):
                    for item in items:
                        run_ref = URIRef(uri+"/run_attributes{}".format(count))
                        g.add((run_ref, RDF.type, b.run_attribute))
                        g.add((uri, b.has_run_attribute, run_ref))
                        if 'TAG' in item.tag:
                            g.add((run_ref, b.has_tag, Literal(item.text)))
                        elif 'VALUE' in i.tag:
                            g.add((run_ref, b.has_value, Literal(item.text)))
                        else:
                            print("Missing in : RUN_ATTRIBUTE", item.tag) 
                    count += 1

print("Storing output here: ", os.path.join(output_path), "as", "test.n3")
# import pdb; pdb.set_trace()
g.serialize(os.path.join(output_path, 'test.n3'),format='n3')
