package cs643.predictionapp;
import java.io.IOException;

import org.apache.spark.ml.PipelineModel;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;

public class Wine_Quality_Prediction {

    public static void main(String[] args) throws IOException {

        SparkSession spark = SparkSession.builder().appName("Pooja_Wine_Quality_Prediction_app").master("local").getOrCreate();    
      
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
        System.out.println("Prediction-app Running");
        PipelineModel pipelineModel = PipelineModel.load(path);
        Dataset<Row> testDataPredict = pipelineModel.transform(testdata).cache();
        System.out.println("Prediction Using " + usingModel + " Model");
        System.out.println();
        System.out.println("TestDataset Metrics \n");
        MulticlassClassificationEvaluator f1Evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("quality").setPredictionCol("prediction").setMetricName("f1");
        System.out.println("F1 score "+f1Evaluator.evaluate(testDataPredict));
        
        MulticlassClassificationEvaluator accEvaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("quality").setPredictionCol("prediction").setMetricName("accuracy");
        System.out.println("Accuracy score "+accEvaluator.evaluate(testDataPredict));
        
        System.out.println("-----------------------------------------------");
    
    }


    
}
