/*
 * This Java source file was generated by the Gradle 'init' task.
 */
package CS643_Programming_Assignment2;

import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.SparkContext;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaRDD;
public class App {
    public String getGreeting() {
        return "Hello World!";
    }
    public static void main(String[] args) {
        System.out.println(new App().getGreeting());
        SparkSession sp = SparkSession.builder().getOrCreate();

        // SparkConf conf = new SparkConf().setAppName("Main");
//        JavaSparkContext sc = new JavaSparkContext(conf);
//        String datapath= '\TrainingDataset.csv';
//        JavaRDD<String> data = sc.textFile(datapath);
        Dataset<Row> training = sp.read().csv("src/main/resources/TrainingDataset.csv");
        LinearRegression lr = new LinearRegression()
                .setMaxIter(10)
                .setRegParam(0.3)
                .setElasticNetParam(0.8);

// Fit the model.
        LinearRegressionModel lrModel = lr.fit(training);

// Print the coefficients and intercept for linear regression.
        System.out.println("Coefficients: "
                + lrModel.coefficients() + " Intercept: " + lrModel.intercept());

// Summarize the model over the training set and print out some metrics.
        LinearRegressionTrainingSummary trainingSummary = lrModel.summary();
        System.out.println("numIterations: " + trainingSummary.totalIterations());
        System.out.println("objectiveHistory: " + Vectors.dense(trainingSummary.objectiveHistory()));
        trainingSummary.residuals().show();
        System.out.println("RMSE: " + trainingSummary.rootMeanSquaredError());
        System.out.println("r2: " + trainingSummary.r2());
    }
}
