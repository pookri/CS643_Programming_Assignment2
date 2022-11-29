package CS643_Programming_Assignment2;
import java.util.Arrays;
// $example on$
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import scala.Tuple2;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.mllib.regression.*;
import org.apache.spark.mllib.*;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import com.univocity.parsers.csv.Csv;

public class JavaLinearRegressionExample {

    public static void main(String[] args) {

        // $example on$
        SparkConf sparkConf = new SparkConf().setAppName("JavaLinearRegressionExample").setMaster("local[1]");
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);
        
        // System.out.println("Spark run complete");

        // SparkConf sparkConf = new SparkConf().setMaster("local").setAppName("JD Word Counter");

        // JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);

        // JavaRDD<String> inputFile = sparkContext.textFile("C:\\Users\\sutha\\VsCodeProjects\\CS643_Programming_Assignment2\\app\\src\\main\\resources\\helloworld.txt");
        // JavaRDD<String> inputFile = jsc.textFile("app\\src\\main\\resources\\TrainDataset.txt");

        // JavaRDD<String> wordsFromFile = inputFile.flatMap(content -> Arrays.asList(content.split(" ")).iterator());
        
        // inputFile.flatMap(null)

        // JavaPairRDD countData = wordsFromFile.mapToPair(t -> new Tuple2(t, 1)).reduceByKey((x, y) -> (int) x + (int) y);

        // countData.saveAsTextFile("app\\src\\main\\resources\\output.txt");
        // Load and parse the data file.
        String datapath = "app\\src\\main\\resources\\TrainDataset.txt";
        
        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(jsc.sc(), datapath).toJavaRDD();
        
        LogisticRegressionModel model = new LogisticRegressionWithLBFGS()
        .setNumClasses(10)
        .run(data.rdd());
        
        JavaPairRDD<Double, Double> predictionAndLabel =
          data.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));

        
        MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabel.rdd());
        double accuracy = metrics.accuracy();
        System.out.println("Accuracy = " + accuracy);
        // // Save and load model
        model.save(jsc.sc(), "app\\src\\main\\models\\LogisticRegressionModel");
        // DecisionTreeModel sameModel = DecisionTreeModel
        //   .load(jsc.sc(), "target/tmp/myDecisionTreeRegressionModel");
        // // $example off$
        jsc.close();
        // JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3});
        // JavaRDD<LabeledPoint> trainingData = splits[0];
        // JavaRDD<LabeledPoint> testData = splits[1];
    
        // SparkSession spark = SparkSession.builder().appName("JavaLinearRegression")
        //         .getOrCreate();
        // Dataset<Row> csvDataset = spark.read().format("csv").option("header", "true")
        //         .load("src/main/resources/TrainingDataset.csv");
        // LinearRegression lr = new LinearRegression()
        //         .setMaxIter(10)
        //         .setRegParam(0.3)
        //         .setElasticNetParam(0.8);
        // org.apache.spark.ml.regression.LinearRegressionModel lrModel = lr.fit(csvDataset);


      }
    
}
