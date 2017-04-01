package org.deeplearning4j.mlp;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Trained 된 Model 을 Loading 하여 Spark Broadcasting 한후
 * Classification 에서 Broacasting  Model 을 이용하여 Unlabel 된 DataSet 을 Predict 함.
 */
public class MnistMLPExampleWithModelBroadcastForPrediction {
    private static final Logger log = LoggerFactory.getLogger(MnistMLPExampleWithModelBroadcastForPrediction.class);

    @Parameter(names = "-useSparkLocal", description = "Use spark local (helper for testing/running without spark submit)", arity = 1)
    private boolean useSparkLocal = true;

    @Parameter(names = "-batchSizePerWorker", description = "Number of examples to fit each worker with")
    private int batchSizePerWorker = 16;

    @Parameter(names = "-numEpochs", description = "Number of epochs for training")
    private int numEpochs = 15;

    public static void main(String[] args) throws Exception {
        new MnistMLPExampleWithModelBroadcastForPrediction().entryPoint(args);
    }

    protected void entryPoint(String[] args) throws Exception {
        //Handle command line arguments
        JCommander jcmdr = new JCommander(this);
        try {
            jcmdr.parse(args);
        } catch (ParameterException e) {
            //User provides invalid input -> print the usage info
            jcmdr.usage();
            try {
                Thread.sleep(500);
            } catch (Exception e2) {
            }
            throw e;
        }

        SparkConf sparkConf = new SparkConf();
        if (useSparkLocal) {
            sparkConf.setMaster("local[*]");
        }
        sparkConf.setAppName("DL4J Spark MLP Example with Prediction");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        //Load the data into memory then parallelize
        //This isn't a good approach in general - but is simple to use for this example
        DataSetIterator iterTest = new MnistDataSetIterator(batchSizePerWorker, true, 12345);
        List<DataSet> testDataList = new ArrayList<>();
        while (iterTest.hasNext()) {
            testDataList.add(iterTest.next());
        }

        // TODO: 실제는 Unlabelled DataSet 을 사용해야 함!!
        JavaRDD<DataSet> unLabelledData = sc.parallelize(testDataList);


        log.info("***** Load the model *****");
        File modelPath = new File("target/mnist-model.zip");
        //Load the model
        MultiLayerNetwork restored = ModelSerializer.restoreMultiLayerNetwork(modelPath);

        // broadcast model.
        Broadcast<MultiLayerNetwork> broadcastModel = sc.broadcast(restored);


        log.info("***** Predict dataset *****");
        unLabelledData.foreach(new Predict(broadcastModel));
    }

    private static class Predict implements VoidFunction<DataSet> {
        private Broadcast<MultiLayerNetwork> broadcastModel;

        public Predict(Broadcast<MultiLayerNetwork> broadcastModel) {
            this.broadcastModel = broadcastModel;
        }


        @Override
        public void call(DataSet dataSet) throws Exception {
            // TODO: This way is correct???????
            int[] predicted = broadcastModel.getValue().predict(dataSet.getFeatures());

            System.out.println("predicted: " + Arrays.toString(predicted));
        }
    }
}
