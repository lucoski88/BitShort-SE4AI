package model;

import org.apache.spark.ml.Model;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.MinMaxScaler;
import org.apache.spark.ml.feature.MinMaxScalerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionSummary;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;

import java.io.File;

public class LinearRegressionTrainer {
    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            System.err.println("Missing arguments");
            System.exit(1);
        }
        String datasetName = args[0];
        if (!new File(datasetName).exists()) {
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
        System.out.println("Creating the dataset");
        String[] features = new String[] {
                "avgClose"
        };
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(features)
                .setOutputCol("features");
        data = assembler.transform(data);

        MinMaxScaler scaler = new MinMaxScaler()
                .setInputCol("features")
                .setOutputCol("scaledFeatures")
                .setMin(0.0)
                .setMax(1.0);

        Dataset<Row>[] split = data.randomSplit(new double[] {70, 30});
        Dataset<Row> trainingSet = split[0];
        MinMaxScalerModel scalerModel = scaler.fit(trainingSet);
        trainingSet = scalerModel.transform(trainingSet);
        Dataset<Row> testSet = split[1];
        testSet = scalerModel.transform(testSet);

        trainingSet.show();

        LinearRegression glr = new LinearRegression();

        glr.setLabelCol("label");
        glr.setFeaturesCol("scaledFeatures");

        LinearRegressionModel model =  glr.fit(trainingSet);

        System.out.println("Training finished");
        Dataset<Row> prediction = model.transform(testSet);
        prediction.show(100);

        LinearRegressionSummary summary = model.evaluate(testSet);


        double mae = summary.meanAbsoluteError();
        double mse = summary.meanSquaredError();
        double rmse = summary.rootMeanSquaredError();
        double r2 = summary.r2();
        System.out.println("Model summary: ");
        System.out.println("MAE: " + mae);
        System.out.println("MSE: " + mse);
        System.out.println("RMSE: " + rmse);
        System.out.println("R2: " + r2);
        System.out.println(model.coefficients());

        try {
            session.close();
        } catch (Exception e) {

        } finally {
            System.out.println("Completed");
        }

        //model.save("C:/Users/lucad/Desktop/models/model");
        //scalerModel.save("C:/Users/lucad/Desktop/models/scaler");
    }
}