package CS643_Programming_Assignment2;

import java.io.IOException;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.VectorIndexer;
import org.apache.spark.ml.feature.VectorIndexerModel;
import org.apache.spark.ml.regression.DecisionTreeRegressor;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import spire.syntax.primitives;

import org.apache.spark.ml.classification.*;


public class Wine_Quality_Training {

    private static class CommonData { 
        VectorAssembler assembler;
        StandardScaler scaler;
        Dataset<Row> trainingData;
    }

    public static Dataset<Row> cleanData(SparkSession spark) {

        Dataset<Row> training = spark.read()
        .option("header", true)
        .option("multiline", true).option("quote", "\"")
        .option("inferSchema", true)
        .option("delimiter", ";")
        .csv( "app\\src\\main\\resources\\traindata.csv" );

        training = training.na().drop().cache();
        String[] headersTrain = training.schema().names();
        for (String name : headersTrain){ 
            training=training.withColumnRenamed(name, name.replace('"', ' ').trim());
        }
        
        return training;

    }

    public static CommonData createPipeline(SparkSession spark){ 
        CommonData commonData = new CommonData();
        Dataset<Row> training=cleanData(spark);
        String [] headersTrain = training.schema().names();
        VectorAssembler assemblerTrain = new VectorAssembler()
        .setInputCols(headersTrain).setOutputCol("features");
        StandardScaler scaler = new StandardScaler().setInputCol("features").setOutputCol("Scaled_feature");
        commonData.assembler = assemblerTrain;
        commonData.scaler = scaler;
        commonData.trainingData = training;
        return commonData;
    }

    public static void logisticRegrssionModel(SparkSession spark) throws IOException{

        CommonData data = createPipeline(spark);

        LogisticRegression logisticRegression = new LogisticRegression();

        logisticRegression.setLabelCol("quality").setFeaturesCol("features");
        Pipeline pipeline = new Pipeline();
        
        pipeline.setStages( new PipelineStage[]{data.assembler,data.scaler,logisticRegression});
        PipelineModel model = pipeline.fit(data.trainingData);
        
    
        model.write().overwrite().save("app\\src\\main\\resources\\outputModel\\LogisticRegression");
        // model.write().overwrite().save("/Data/dm/outputModel/logisticRegression");
    }

    public static void DecisionTreeModel(SparkSession spark) throws IOException{

        CommonData data = createPipeline(spark);

        DecisionTreeClassifier decisionTreeClassifier = new DecisionTreeClassifier();

        decisionTreeClassifier.setLabelCol("quality").setFeaturesCol("features");
        Pipeline pipeline = new Pipeline();
        
        pipeline.setStages( new PipelineStage[]{data.assembler,data.scaler,decisionTreeClassifier});
        PipelineModel model = pipeline.fit(data.trainingData);

        model.write().overwrite().save("app\\src\\main\\resources\\outputModel\\DecisionTreeClassifier");
       
        // model.write().overwrite().save("/Data/dm/outputModel/DecisionTree");
    }

    public static void NavieBayesModel(SparkSession spark) throws IOException{

        CommonData data = createPipeline(spark);

        NaiveBayes naiveBayes = new NaiveBayes();

        naiveBayes.setLabelCol("quality").setFeaturesCol("features");
        Pipeline pipeline = new Pipeline();
        
        pipeline.setStages( new PipelineStage[]{data.assembler,data.scaler,naiveBayes});
        PipelineModel model = pipeline.fit(data.trainingData);

        model.write().overwrite().save("app\\src\\main\\resources\\outputModel\\NaiveBayesClassifier");
        
        // model.write().overwrite().save("/Data/dm/outputModel/DecisionTree");
    }

    public static void RandomForestModel(SparkSession spark) throws IOException{

        CommonData data = createPipeline(spark);

        RandomForestClassifier rClassifier = new RandomForestClassifier();

        rClassifier.setLabelCol("quality").setFeaturesCol("features");
        Pipeline pipeline = new Pipeline();
        
        pipeline.setStages( new PipelineStage[]{data.assembler,data.scaler,rClassifier});
        PipelineModel model = pipeline.fit(data.trainingData);

        model.write().overwrite().save("app\\src\\main\\resources\\outputModel\\RandomForestClassifier");
        
        // model.write().overwrite().save("/Data/dm/outputModel/DecisionTree");
    }

    public static void DecisionTreeRegressorModel(SparkSession spark) throws IOException{

        CommonData data = createPipeline(spark);

        DecisionTreeRegressor decisionTreeRegressor = new DecisionTreeRegressor();

        decisionTreeRegressor.setLabelCol("quality").setFeaturesCol("features");
        Pipeline pipeline = new Pipeline();
        
        pipeline.setStages( new PipelineStage[]{data.assembler,data.scaler,decisionTreeRegressor});
        PipelineModel model = pipeline.fit(data.trainingData);

        model.write().overwrite().save("app\\src\\main\\resources\\outputModel\\DecisionTreeRegressor");
        // model.write().overwrite().save("/Data/dm/outputModel/DecisionTree");
    }

    public static void main(String[] args) throws IOException {
        System.out.println("Wine Quality Training");
        SparkSession spark = SparkSession.builder().appName("JavaLinerRegressionModel").master("local").getOrCreate();    
        logisticRegrssionModel(spark);
        DecisionTreeModel(spark);
        // NavieBayesModel(spark);
        RandomForestModel(spark);
        DecisionTreeRegressorModel(spark);
        spark.stop();
    }


    
}
