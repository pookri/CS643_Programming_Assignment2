package CS643_Programming_Assignment2;

import java.io.IOException;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
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
        .csv( "traindata.csv" );

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
        
        // LogisticRegressionModel lrModel = (LogisticRegressionModel)(model.stages()[2]);
        model.write().overwrite().save("logisticRegression");
    }

    public static void main(String[] args) throws IOException {
        System.out.println("Wine Quality Training");
        SparkSession spark = SparkSession.builder().appName("JavaLinerRegressionModel").master("local").getOrCreate();    
        logisticRegrssionModel(spark);
        spark.stop();
    }


    
}
