package model;

import org.apache.spark.ml.*;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.MinMaxScaler;
import org.apache.spark.ml.feature.MinMaxScalerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.DoubleArrayParam;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionSummary;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.scheduler.Stage;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.StructType;

import java.io.File;
import java.util.Arrays;

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
        StructType schema = data.schema();
        String[] fieldNames = schema.fieldNames();
        String[] features = new String[fieldNames.length - 1];
        System.arraycopy(fieldNames, 0, features, 0, features.length);

        System.out.println("Starting Linear Regression training");
        Dataset<Row>[] split = data.randomSplit(new double[] {0.7, 0.3});
        Dataset<Row> trainingSet = split[0];
        Dataset<Row> testSet = split[1];

        VectorAssembler featuresAssembler = new VectorAssembler()
                .setInputCols(features)
                .setOutputCol("features");
        MinMaxScaler featuresScaler = new MinMaxScaler()
                .setInputCol("features")
                .setOutputCol("scaledFeatures")
                .setMin(0.0)
                .setMax(1.0);
        LinearRegression regression = new LinearRegression();
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{featuresAssembler, featuresScaler, regression});

        ParamGridBuilder paramGridBuilder = new ParamGridBuilder();
        ParamMap[] paramGrid = paramGridBuilder
                .addGrid(regression.regParam(), new double[] {0, 0.5, 0.9})
                .addGrid(regression.elasticNetParam(), new double[] {0, 0.5, 0.9})
                .addGrid(regression.maxIter(), new int[] {10, 25, 50, 100})
                //.addGrid(regression.epsilon(), new double[] {0.5, 0.9})
                .addGrid(regression.fitIntercept())
                .build();
        RegressionEvaluator maeEvaluator = new RegressionEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("mae");
        RegressionEvaluator mseEvaluator = new RegressionEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("mse");
        RegressionEvaluator rmseEvaluator = new RegressionEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("rmse");
        RegressionEvaluator r2Evaluator = new RegressionEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("r2");
        CrossValidator crossValidator = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(maeEvaluator)
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(5)
                .setParallelism(2);
        System.out.println("Fitting CrossValidator...");
        CrossValidatorModel cvModel = crossValidator.fit(trainingSet);
        System.out.println("CrossValidator fit completed");
        PipelineModel bestModel = (PipelineModel) cvModel.bestModel();
        System.out.println("Evaluating best model on test set...");
        Dataset<Row> predictions = bestModel.transform(testSet);
        double mae = maeEvaluator.evaluate(predictions);
        double mse = mseEvaluator.evaluate(predictions);
        double rmse = rmseEvaluator.evaluate(predictions);
        double r2 = r2Evaluator.evaluate(predictions);
        System.out.println("Evaluation metrics: ");
        System.out.println("\tMAE: " + mae + ", is larger better: " + maeEvaluator.isLargerBetter());
        System.out.println("\tMSE: " + mse + ", is larger better: " + mseEvaluator.isLargerBetter());
        System.out.println("\tRMSE: " + rmse + ", is larger better: " + rmseEvaluator.isLargerBetter());
        System.out.println("\tR2: " + r2 + ", is larger better: " + r2Evaluator.isLargerBetter());
        System.out.println("Model Coefficients:");
        LinearRegressionModel model = null;
        for (PipelineStage s : ((PipelineModel) cvModel.bestModel()).stages())  {
            if (s instanceof LinearRegressionModel) {
                model = (LinearRegressionModel) s;
                break;
            }
        }
        System.out.println("\t" + model.coefficients());
        System.out.println("Best hyperparameters:");
        System.out.println("\tRegParam: " + model.getRegParam());
        System.out.println("\tElasticNetParam: " + model.getElasticNetParam());
        System.out.println("\tMaxIter: " + model.getMaxIter());
        System.out.println("\tEpsilon: " + model.getEpsilon());
        System.out.println("\tLoss: " + model.getLoss());
        System.out.println("\tFitIntercept: " + model.getFitIntercept());
        session.stop();
        System.exit(0);

        //model.save("C:/Users/lucad/Desktop/models/model");
        //scalerModel.save("C:/Users/lucad/Desktop/models/scaler");
    }
}