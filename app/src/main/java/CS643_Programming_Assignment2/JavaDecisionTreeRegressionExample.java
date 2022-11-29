package CS643_Programming_Assignment2;
// $example on$
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import scala.Tuple2;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class JavaDecisionTreeRegressionExample {
    
    public static void main(String[] args) {

        // $example on$
        SparkConf sparkConf = new SparkConf().setAppName("JavaDecisionTreeRegressionExample");
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);
    
        // Load and parse the data file.
        String datapath = "src/main/resources/TrainingDataset.csv";
        SparkSession spark = SparkSession.builder().appName("CDX JSON Merge Job")
                .getOrCreate();
        Dataset<Row> csvDataset = spark.read().format("csv").option("header", "true")
                .load("C:\\sample.csv");
        csvDataset.createOrReplaceTempView("csvdataTable");
        Dataset<Row> reducedCSVDataset = spark.sql("select VendorID from csvdataTable limit 2 ");
        Dataset<String> rdds = reducedCSVDataset.toDF().select("VendorID").as(Encoders.STRING());
        List<String> listOfStrings = rdds.collectAsList();
        listOfStrings.forEach(x -> System.out.println(x));
        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(jsc.sc(), datapath).toJavaRDD();
        // Split the data into training and test sets (30% held out for testing)
        JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3});
        JavaRDD<LabeledPoint> trainingData = splits[0];
        JavaRDD<LabeledPoint> testData = splits[1];
    
        // Set parameters.
        // Empty categoricalFeaturesInfo indicates all features are continuous.
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        String impurity = "variance";
        int maxDepth = 5;
        int maxBins = 32;
    
        // Train a DecisionTree model.
        DecisionTreeModel model = DecisionTree.trainRegressor(trainingData,
          categoricalFeaturesInfo, impurity, maxDepth, maxBins);
    
        // Evaluate model on test instances and compute test error
        JavaPairRDD<Double, Double> predictionAndLabel =
          testData.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));
        double testMSE = predictionAndLabel.mapToDouble(pl -> {
          double diff = pl._1() - pl._2();
          return diff * diff;
        }).mean();
        System.out.println("Test Mean Squared Error: " + testMSE);
        System.out.println("Learned regression tree model:\n" + model.toDebugString());
    
        // Save and load model
        model.save(jsc.sc(), "target/tmp/myDecisionTreeRegressionModel");
        DecisionTreeModel sameModel = DecisionTreeModel
          .load(jsc.sc(), "target/tmp/myDecisionTreeRegressionModel");
        // $example off$
        jsc.close();
      }
}
