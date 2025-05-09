package model;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.MinMaxScaler;
import org.apache.spark.ml.feature.MinMaxScalerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.*;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructType;

import java.io.File;

public class RandomForestRegressorTrainer {
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

        System.out.println("Starting Random Forest Regressor training");
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

        RandomForestRegressor regression = new RandomForestRegressor();
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{featuresAssembler, featuresScaler, regression});

        ParamGridBuilder paramGridBuilder = new ParamGridBuilder();
        ParamMap[] paramGrid = paramGridBuilder
                .addGrid(regression.numTrees(), new int[] {20, 10, 30})
                .addGrid(regression.maxDepth(), new int[] {5, 2, 10})
                .addGrid(regression.maxBins(), new int[] {32, 16, 64})
                .addGrid(regression.minInstancesPerNode(), new int[] {1, 2, 3})
                .addGrid(regression.minInfoGain(), new double[] {0.0})
                .addGrid(regression.subsamplingRate(), new double[] {1.0})
                //.addGrid(regression.featureSubsetStrategy(), new String[] {})
                //.addGrid(regression.impurity(), new String[] {})
                .addGrid(regression.bootstrap())
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
        RandomForestRegressionModel model = null;
        for (PipelineStage s : ((PipelineModel) cvModel.bestModel()).stages())  {
            if (s instanceof RandomForestRegressionModel) {
                model = (RandomForestRegressionModel) s;
                break;
            }
        }
        System.out.println("Best hyperparameters:");
        System.out.println("\tNumTrees: " + model.getNumTrees());
        System.out.println("\tMaxDepth: " + model.getMaxDepth());
        System.out.println("\tMaxBins: " + model.getMaxBins());
        System.out.println("\tMinInstancesPerNode: " + model.getMinInstancesPerNode());
        System.out.println("\tMinInfoGain: " + model.getMinInfoGain());
        System.out.println("\tSubSamplingRate: " + model.getSubsamplingRate());
        System.out.println("\tFeatureSubsetStrategy: " + model.getFeatureSubsetStrategy());
        System.out.println("\tImpurity: " + model.getImpurity());
        System.out.println("\tBootstrap: " + model.getBootstrap());
        System.out.println("\tSeed: " + model.getSeed());
        session.stop();
        System.exit(0);

        //model.save("C:/Users/lucad/Desktop/models/model");
        //scalerModel.save("C:/Users/lucad/Desktop/models/scaler");
    }
}