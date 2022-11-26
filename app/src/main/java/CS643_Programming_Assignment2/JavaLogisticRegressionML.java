package CS643_Programming_Assignment2;

import java.io.IOException;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.VectorAssembler;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;



public class JavaLogisticRegressionML {

    public static void main(String[] args) throws IOException {

        SparkSession spark = SparkSession.builder().appName("JavaLinerRegressionModel").master("local").getOrCreate();    

        
        Dataset<Row> training = spark.read()
        .option("header", true)
        .option("inferSchema", true)
        .option("delimiter", ",")
        .csv( "app\\src\\main\\resources\\TrainingDataset.csv" );

        Dataset<Row> validation = spark.read()
        .option("header", true)
        .option("inferSchema", true)
        .option("delimiter", ",")
        .csv( "app\\src\\main\\resources\\ValidationDataset.csv" );

        String[] headersTrain = training.schema().names();
         
        VectorAssembler assemblerTrain = new VectorAssembler()
        .setInputCols(headersTrain).setOutputCol("features");
        StandardScaler scaler = new StandardScaler().setInputCol("features").setOutputCol("Scaled_feature");

        LogisticRegression logisticRegression = new LogisticRegression();
      
        logisticRegression.setLabelCol("quality").setFeaturesCol("features");
        Pipeline pipeline = new Pipeline();
        
        pipeline.setStages( new PipelineStage[]{assemblerTrain,scaler,logisticRegression} );
        PipelineModel model = pipeline.fit(training);
        
        Dataset<Row> validationPredict = model.transform(validation);
        
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator().setLabelCol("quality").setPredictionCol("prediction");
    
        MulticlassClassificationEvaluator f1Evaluator = evaluator.setMetricName("f1");
        System.out.println("F1 score "+f1Evaluator.evaluate(validationPredict));

        MulticlassClassificationEvaluator accEvaluator = evaluator.setMetricName("accuracy");
        System.out.println("Accuracy score "+accEvaluator.evaluate(validationPredict));
        
        
        // model.write().save("app\\src\\main\\resources\\outputModel\\logisticRegression");
       
        spark.stop();
    }
    
}
