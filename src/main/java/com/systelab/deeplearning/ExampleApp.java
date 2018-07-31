package com.systelab.deeplearning;

public class ExampleApp {
    public static void main(String[] args) {
        IrisClassifier classifier = new IrisClassifier();
        try {
            classifier.classify("iris.csv","iris-test.csv");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
