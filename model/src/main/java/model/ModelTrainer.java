package model;

import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.File;

public class ModelTrainer {
    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            System.err.println("Missing arguments");
            System.exit(1);
        }
        String datasetName = args[0];
        if (new File(datasetName).exists()) {
            System.err.println(datasetName + " not found");
            System.exit(1);
        }

        SparkSession session = SparkSession
                .builder()
                .appName("ModelCreator")
                .master("local")
                .getOrCreate();

        Dataset<Row> data = session.read().format("csv")
                .option("header", true)
                .option("inferSchema", true)
                .load(datasetName);
        Dataset<Row> selectedData = data;

        selectedData.show(10);

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"open", "low", "actual_open"})
                .setOutputCol("features");

        selectedData = assembler.transform(selectedData);

        LinearRegression glr = new LinearRegression();

        glr.setLoss("huber");

        glr.setLabelCol("target");
        glr.setFeaturesCol("features");

        LinearRegressionModel model =  glr.fit(selectedData);
        model.save("model");
        System.out.println("Training finished");

        Dataset<Row> testData = session.read().format("csv")
                .option("header", true)
                .option("inferSchema", true)
                .load("from_binance_last_week_dataset.csv");

        testData = assembler.transform(testData);

        Dataset<Row> prediction = model.transform(testData);
        prediction.show();

        RegressionEvaluator maeEvaluator = new RegressionEvaluator()
                .setLabelCol("target")
                .setPredictionCol("prediction")
                .setMetricName("mae");
        RegressionEvaluator mseEvaluator = new RegressionEvaluator()
                .setLabelCol("target")
                .setPredictionCol("prediction")
                .setMetricName("mse");
        RegressionEvaluator rmseEvaluator = new RegressionEvaluator()
                .setLabelCol("target")
                .setPredictionCol("prediction")
                .setMetricName("rmse");
        RegressionEvaluator r2Evaluator = new RegressionEvaluator()
                .setLabelCol("target")
                .setPredictionCol("prediction")
                .setMetricName("r2");

        double mae = maeEvaluator.evaluate(prediction);
        double mse = mseEvaluator.evaluate(prediction);
        double rmse = rmseEvaluator.evaluate(prediction);
        double r2 = r2Evaluator.evaluate(prediction);
        System.out.println("LinearRegeressionModel metrics:");
        System.out.println("MAE: " + mae +  " IsLargerBetter? " + maeEvaluator.isLargerBetter());
        System.out.println("MSE: " + mse +  " IsLargerBetter? " + mseEvaluator.isLargerBetter());
        System.out.println("RMSE: " + rmse +  " IsLargerBetter? " + rmseEvaluator.isLargerBetter());
        System.out.println("R2: " + r2 +  " IsLargerBetter? " + r2Evaluator.isLargerBetter());
    }
}