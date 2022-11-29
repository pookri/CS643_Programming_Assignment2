package CS643_Programming_Assignment2;

import org.apache.spark.ml.regression.DecisionTreeRegressionModel;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.ml.stat.MultiClassSummarizer;

import java.io.IOException;
import java.lang.reflect.Array;
import java.util.Arrays;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionSummary;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.netlib.blas.Dcopy;

import breeze.stats.mode;

public class JavaLinearRegressionML {

    public static void main(String[] args) throws IOException {

        SparkSession spark = SparkSession.builder().appName("JavaLinerRegressionModel").master("local").getOrCreate();    
        // SparkSession spark = SparkSession.builder().appName("MyRegModel").config("spark.master", "local").getOrCreate();
        
        // VectorAssembler assembler = new VectorAssembler();
        // assembler.setInputCols( (String[]) Arrays.asList("fixed acidity","volatile acidity","citric acid",
        // "residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH",
        // "sulphates","alcohol","quality").toArray())
        // .setOutputCol("features");
        
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

        Dataset<Row>[] splits = training.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        // validation = validation.select("fixed acidity","volatile acidity","citric acid",
        // "residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH",
        // "sulphates","alcohol");
        String[] headersTrain = trainingData.schema().names();
        
        // for (String name : headersTrain){ 
        //     training.withColumnRenamed(name, name.replace('"', ' ').trim());
        // }
        
        VectorAssembler assemblerTrain = new VectorAssembler()
        .setInputCols(headersTrain).setOutputCol("features");
        StandardScaler scaler = new StandardScaler().setInputCol("features").setOutputCol("Scaled_feature");

        LogisticRegression logisticRegression = new LogisticRegression();

        DecisionTreeClassifier dClassifier = new DecisionTreeClassifier();
        dClassifier.setLabelCol("quality").setFeaturesCol("features");
        
        // LinearRegression logisticRegression = new LinearRegression();
        logisticRegression.setLabelCol("quality").setFeaturesCol("features");
        Pipeline pipeline = new Pipeline();
        
        pipeline.setStages( new PipelineStage[]{assemblerTrain, dClassifier} );
        PipelineModel model = pipeline.fit(trainingData);
        
        Dataset<Row> validationPredict = model.transform(testData);
        
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("quality").setPredictionCol("prediction").setMetricName("accuracy");
        Double acc = evaluator.evaluate(validationPredict);
        // evaluator.evaluate(validationPredict);
        // MulticlassClassificationEvaluator f1Evaluator = evaluator.setMetricName("f1");
        // System.out.println("F1 score "+f1Evaluator.evaluate(validationPredict));

        // MulticlassClassificationEvaluator accEvaluator = evaluator.setMetricName("accuracy");
        System.out.println("Accuracy score "+acc);

        
        // model.write().save("app\\src\\main\\resources\\outputModel\\logisticRegression");
        // training.withColumnRenamed(null, null)
        
        // LinearRegression lr = new LinearRegression().setMaxIter(10)
        // .setRegParam(0.3)
        // .setElasticNetParam(0.8);
    
        // LogisticRegression regression = new LogisticRegression()
        // .setFeaturesCol("features");

            
        // Pipeline pipeline = new Pipeline().setStages((PipelineStage[]) Arrays.asList(assembler, regression).toArray());
        // // Fit the model.
        // LinearRegressionModel lrModel = lr.fit(training);
        // pipeline.fit(training);

        // // Print the coefficients and intercept for linear regression.
        // System.out.println("Coefficients: " + lrModel.coefficients() + " Intercept: " + lrModel.intercept());

        // // Summarize the model over the training set and print out some metrics.
        // LinearRegressionTrainingSummary trainingSummary = lrModel.summary();
        // System.out.println("numIterations: " + trainingSummary.totalIterations());
        // System.out.println("numIterations: " + trainingSummary.totalIterations());
        // System.out.println("objectiveHistory: " + Vectors.dense(trainingSummary.objectiveHistory()));
        // trainingSummary.residuals().show();
        // System.out.println("RMSE: " + trainingSummary.rootMeanSquaredError());
        // System.out.println("r2: " + trainingSummary.r2());
        spark.stop();
    }


}
