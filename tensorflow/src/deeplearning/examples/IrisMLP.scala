package deeplearning.examples

import com.typesafe.scalalogging.Logger
import org.platanios.tensorflow.api.ops.io.data.Dataset
import org.platanios.tensorflow.api.types.DataType
import org.platanios.tensorflow.api.{tf, _}
import org.slf4j.LoggerFactory

object IrisMLP {

  private[this] val logger = Logger(LoggerFactory.getLogger("tensorflow/Examples/Iris"))

  def main(args: Array[String]): Unit = {

    val seed         = 1
    val numInputs    = 4
    val numHidden    = 10
    val numOutputs   = 3
    val learningRate = 0.1
    val iterations   = 1000
    val trainSize    = 100
    val testSize     = 50

    // Read CSV using TensorFlow operations
    val dataset: Dataset[(Tensor, Tensor), (Output, Output), (DataType, DataType), (Shape, Shape)] =
      tf.data.TextLinesDataset("tensorflow/data/iris.csv")
        .drop(1)
        .map { l =>
          val csv = tf.decodeCSV(l, Seq.fill(5)(Tensor(FLOAT32)), Seq.fill(5)(FLOAT32))
          (tf.stack(csv.take(4)), csv(4))
        }
        .shuffle(150, Some(seed))

    //Dividing the dataset into train and test
    val trainDataset = dataset.take(trainSize)
    val testDataset = dataset.drop(trainSize)

    val trainData = trainDataset.repeat().batch(trainSize)
    val evalTrainData = trainDataset.batch(trainSize)
    val evalTestData = testDataset.batch(testSize)


    val input = tf.learn.Input(FLOAT32, Shape(-1, trainDataset.outputShapes._1(0)))
    val trainInput = tf.learn.Input(FLOAT32, Shape(-1))
    val layer = tf.learn.Linear("Layer_0/Linear", numHidden) >> tf.learn.ReLU("Layer_0/ReLU") >>
      tf.learn.Linear("OutputLayer/Linear", numOutputs)
    val trainingInputLayer = tf.learn.Cast("TrainInput/Cast", INT64)
    val loss = tf.learn.SparseSoftmaxCrossEntropy("Loss/CrossEntropy") >>
      tf.learn.Mean("Loss/Mean") >> tf.learn.ScalarSummary("Loss/Summary", "Loss")
    val optimizer = tf.train.GradientDescent(learningRate)
    val model = tf.learn.Model.supervised(input, layer, trainInput, trainingInputLayer, loss, optimizer)

    val summariesDir = Paths.get("tensorflow/temp/iris-mlp")
    val accMetric = tf.metrics.MapMetric(
      (v: (Output, Output)) => (v._1.argmax(1), v._2), tf.metrics.Accuracy())
    val estimator = tf.learn.InMemoryEstimator(
      model,
      tf.learn.Configuration(Some(summariesDir)),
      tf.learn.StopCriteria(maxSteps = Some(iterations)),
      Set(
        tf.learn.LossLogger(trigger = tf.learn.StepHookTrigger(100)),
        tf.learn.Evaluator(
          log = true, datasets = Seq(("Train", () => evalTrainData), ("Test", () => evalTestData)),
          metrics = Seq(accMetric), trigger = tf.learn.StepHookTrigger(1000), name = "Evaluator"),
        tf.learn.StepRateLogger(log = false, summaryDir = summariesDir, trigger = tf.learn.StepHookTrigger(100)),
        tf.learn.SummarySaver(summariesDir, tf.learn.StepHookTrigger(100)),
        tf.learn.CheckpointSaver(summariesDir, tf.learn.StepHookTrigger(100))),
      tensorBoardConfig = tf.learn.TensorBoardConfig(summariesDir, reloadInterval = 1))

    estimator.train(() => trainData)

  }

}
