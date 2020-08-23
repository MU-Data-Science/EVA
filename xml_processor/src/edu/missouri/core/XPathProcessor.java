package edu.missouri.core;

import edu.missouri.util.XPathUtil;

import java.util.List;
import org.w3c.dom.Document;

public class XPathProcessor {
    public static void main(String[] args) {
        if(args != null) {
            try {
                String xml = args[0];
                String xpathQuery = args[1];

                // Convert XML String to a Document.
                Document xmlDocument = XPathUtil.getInstance().convertStringToDocument(xml);

                // Performing XPath querying on the XML Document.
                List<String> results = XPathUtil.getInstance().executeXPathQuery(xmlDocument, xpathQuery);

                System.out.println(results);
            } catch (Exception e) {
                usage();
            }
        } else {
            usage();
        }
    }

    public static void usage() {
        System.out.println("java -cp XMLProcessor.jar edu.missouri.core.XPathProcessor '<XML>' '<XPATH>'");
    }


}
