package edu.missouri.util;

import org.w3c.dom.Document;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.xpath.*;

import org.w3c.dom.NodeList;
import org.xml.sax.InputSource;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;

public class XPathUtil {
    public static XPathUtil instance = null;

    private XPathUtil() {

    }

    public static XPathUtil getInstance() {
        if (instance == null) {
            instance = new XPathUtil();
        }
        return instance;
    }

    public static Document convertStringToDocument(String xmlStr) {
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        DocumentBuilder builder;
        try {
            builder = factory.newDocumentBuilder();
            Document doc = builder.parse(new InputSource(new StringReader(xmlStr)));
            return doc;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    public static List<String> executeXPathQuery(Document doc, String query) {
        // Creating XPathFactory object
        XPathFactory xpathFactory = XPathFactory.newInstance();

        // Creating XPath object
        XPath xpath = xpathFactory.newXPath();

        List<String> list = new ArrayList<String>();
        try {
            // Executing the Xpath query.
            XPathExpression expr = xpath.compile(query);
            NodeList nodes = (NodeList) expr.evaluate(doc, XPathConstants.NODESET);
            for (int i = 0; i < nodes.getLength(); i++) {
                //Adding the contents to the list.
                list.add(nodes.item(i).getTextContent());
            }
        } catch (XPathExpressionException e) {
            e.printStackTrace();
        }
        return list;
    }
}
