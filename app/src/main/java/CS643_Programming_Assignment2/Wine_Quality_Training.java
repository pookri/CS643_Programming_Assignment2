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
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;


public class Wine_Quality_Training {

    private static class CommonData { 
        VectorAssembler assemblerTrain;
        StandardScaler scalerTrain;
        Dataset<Row> trainingData;
        Dataset<Row> validData;
    }

    public static Dataset<Row> cleanTrainData(SparkSession spark, boolean printTable) {

        Dataset<Row> training = spark.read()
        .option("header", true)
        .option("multiline", true).option("quote", "\"")
        .option("inferSchema", true)
        .option("delimiter", ";")
        .csv( "TrainingDataset.csv" );

        training = training.na().drop().cache();
        String[] headersTrain = training.schema().names();
        for (String name : headersTrain){ 
            training=training.withColumnRenamed(name, name.replace('"', ' ').trim());
        }
        training=training.withColumnRenamed("quality", "label").select("label",
        "alcohol", "sulphates", "pH", "density", "free sulfur dioxide", "total sulfur dioxide",
        "chlorides", "residual sugar", "citric acid", "volatile acidity", "fixed acidity");
        
        if (printTable){
            System.out.println("Training Dataset \n");
            training.show(5);
        }
        
        return training;

    }

    public static Dataset<Row> cleanValidData(SparkSession spark, boolean printTable) {

        Dataset<Row> validationDataset = spark.read()
        .option("header", true)
        .option("multiline", true).option("quote", "\"")
        .option("inferSchema", true)
        .option("delimiter", ";")
        .csv( "ValidationDataset.csv" );

        validationDataset = validationDataset.na().drop().cache();
        String[] headersTrain = validationDataset.schema().names();
        for (String name : headersTrain){ 
            validationDataset=validationDataset.withColumnRenamed(name, name.replace('"', ' ').trim());
        }
        validationDataset=validationDataset.withColumnRenamed("quality", "label").select("label",
        "alcohol", "sulphates", "pH", "density", "free sulfur dioxide", "total sulfur dioxide",
        "chlorides", "residual sugar", "citric acid", "volatile acidity", "fixed acidity");
        
        if(printTable) {
            System.out.println("Validation Dataset \n");
            validationDataset.show(5);
        }
        
        return validationDataset;

    }

    public static CommonData createPipeline(SparkSession spark){ 
        CommonData commonData = new CommonData();
        
        Dataset<Row> training=cleanTrainData(spark, false);
        Dataset<Row> valiDataset = cleanValidData(spark, false);
        
        String [] headersTrain = training.schema().names();
        VectorAssembler assemblerTrain = new VectorAssembler()
        .setInputCols(headersTrain).setOutputCol("features");
        training=assemblerTrain.transform(training).select("label","features");
        valiDataset=assemblerTrain.transform(valiDataset).select("label","features");
        StandardScaler scalerTrain = new StandardScaler().setInputCol("features").setOutputCol("Scaled_feature");
        
        commonData.assemblerTrain = assemblerTrain;
        commonData.scalerTrain = scalerTrain;
        commonData.trainingData = training;
        commonData.validData = valiDataset;

        return commonData;
    }

    public static void printMertics(Dataset<Row> predictions) {
        System.out.println();
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator();
        evaluator.setMetricName("accuracy");
        System.out.println("The accuracy of the model is " + evaluator.evaluate(predictions));
        evaluator.setMetricName("f1");
        double f1 = evaluator.evaluate(predictions);
        System.out.println("F1: " + f1);
    }

    public static void logisticRegrssionModel(SparkSession spark) throws IOException{

        CommonData data = createPipeline(spark);

        LogisticRegression logisticRegression = new LogisticRegression();

        logisticRegression.setLabelCol("label").setFeaturesCol("features");
        Pipeline pipeline = new Pipeline();
        
        pipeline.setStages( new PipelineStage[]{logisticRegression});
        PipelineModel model = pipeline.fit(data.trainingData);
        
        LogisticRegressionModel lrModel = (LogisticRegressionModel) (model.stages()[0]);

        LogisticRegressionTrainingSummary trainingSummary = lrModel.summary();
        double accuracy = trainingSummary.accuracy();
        double fMeasure = trainingSummary.weightedFMeasure();

        System.out.println();
        System.out.println("Logistic Regression Model \n");
        System.out.println("Training DataSet Metrics ");

        System.out.println("Accuracy: " + accuracy);
        System.out.println("F-measure: " + fMeasure);

        Dataset<Row> validDataPredict = model.transform(data.validData).cache();
        
        System.out.println("\n Validation Training Set Metrics");
        validDataPredict.select("features", "label", "prediction").show(5, false);
        printMertics(validDataPredict);


        model.write().overwrite().save("LogisticRegression");
    }

    public static void DecisionTreeModel(SparkSession spark) throws IOException{

        CommonData data = createPipeline(spark);

        DecisionTreeClassifier decisionTreeClassifier = new DecisionTreeClassifier();

        decisionTreeClassifier.setLabelCol("label").setFeaturesCol("features");
        Pipeline pipeline = new Pipeline();
        
        pipeline.setStages( new PipelineStage[]{decisionTreeClassifier});
        PipelineModel model = pipeline.fit(data.trainingData);

        Dataset<Row> validDataPredict = model.transform(data.validData).cache();
        System.out.println("Decision Tree Model \n");
        System.out.println("\n Validation Training Set Metrics");
        validDataPredict.select("features", "label", "prediction").show(5, false);
        printMertics(validDataPredict);

        model.write().overwrite().save("DecisionTreeClassifier");
       
    }

    public static void RandomForestModel(SparkSession spark) throws IOException{

        CommonData data = createPipeline(spark);

        RandomForestClassifier rClassifier = new RandomForestClassifier();

        rClassifier.setLabelCol("label").setFeaturesCol("features");
        Pipeline pipeline = new Pipeline();

        pipeline.setStages( new PipelineStage[]{rClassifier});
        PipelineModel model = pipeline.fit(data.trainingData);

        RandomForestClassificationModel rfModel = (RandomForestClassificationModel) (model.stages()[0]);

        RandomForestClassificationSummary trainingSummary = rfModel.summary();
        double accuracy = trainingSummary.accuracy();
        double fMeasure = trainingSummary.weightedFMeasure();

        System.out.println();
        System.out.println("Random Forest Model \n");
        System.out.println("Training DataSet Metrics ");

        System.out.println("Accuracy: " + accuracy);
        System.out.println("F-measure: " + fMeasure);

        Dataset<Row> validDataPredict = model.transform(data.validData).cache();
        System.out.println("\n Validation Training Set Metrics");
        validDataPredict.select("features", "label", "prediction").show(5, false);
        printMertics(validDataPredict);

        // model.write().overwrite().save("app\\src\\main\\resources\\Models\\RandomForestClassifier");
        model.write().overwrite().save("RandomForestClassifier");
    
    }

    public static void main(String[] args) throws IOException {
        System.out.println("Wine Quality Training");
        SparkSession spark = SparkSession.builder().appName("Pooja_Wine_Quality_Training_App").master("local").getOrCreate();    
        cleanTrainData(spark, true);
        cleanValidData(spark, true);
        logisticRegrssionModel(spark);
        DecisionTreeModel(spark);
        RandomForestModel(spark);
        spark.stop();
    }
    
}
