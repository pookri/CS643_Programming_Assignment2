package cs643.predictionapp;
import java.io.IOException;

import org.apache.spark.ml.PipelineModel;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;

public class Wine_Quality_Prediction {

    public static void main(String[] args) throws IOException {

        SparkSession spark = SparkSession.builder().appName("JavaLinerRegressionModel").master("local").getOrCreate();    
      
        Dataset<Row> testdata = spark.read()
        .option("header", true)
        .option("multiline", true).option("quote", "\"")
        .option("inferSchema", true)
        .option("delimiter", ";")
        .csv( "/Data/ValidationDataset.csv" );

        testdata = testdata.na().drop().cache();

        String[] headersTrain = testdata.schema().names();
        
        for (String name : headersTrain){ 
            testdata=testdata.withColumnRenamed(name, name.replace('"', ' ').trim());
        }

        printModel("/Models/LogisticRegression", testdata, "Logistic Regression");
        printModel("/Models/DecisionTreeClassifier", testdata, "Descision Tree Classifier");        
        printModel("/Models/RandomForestClassifier", testdata, "Random Forest Classifier");
 
        spark.stop();
    }

    private static void printModel(String path, Dataset<Row> testdata, String usingModel) { 
        PipelineModel pipelineModel = PipelineModel.load(path);
        Dataset<Row> validationPredict = pipelineModel.transform(testdata);
        System.out.println("Prediction Using Model " + usingModel);
        MulticlassClassificationEvaluator f1Evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("quality").setPredictionCol("prediction").setMetricName("f1");
        System.out.println("F1 score "+f1Evaluator.evaluate(validationPredict));
        
        MulticlassClassificationEvaluator accEvaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("quality").setPredictionCol("prediction").setMetricName("accuracy");
        System.out.println("Accuracy score "+accEvaluator.evaluate(validationPredict));
        
        System.out.println("-----------------------------------------------");
    
    }


    
}
