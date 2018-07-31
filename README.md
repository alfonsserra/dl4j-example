
# DeepLearning 4J Example

This project is an example of a classification using DeepLearning 4J. It is based in the post [A Quick Easy Guide to Deep Learning with Java][opencodez]

## Getting Started

To get you started you can simply clone the `dl4j-example` repository and install the dependencies:

### Prerequisites

You need [git][git] to clone the `dl4j-example` repository.

You will need [Java™ SE Development Kit 8][jdk-download] and [Maven][maven].


### Clone `dl4j-example`

Clone the `dl4j-example` repository using git:

```bash
git clone https://github.com/systelab/dl4j-example.git
cd dl4j-example
```

If you just want to start a new project without the `dl4j-example` commit history then you can do:

```bash
git clone --depth=1 https://github.com/systelab/dl4j-example.git <your-project-name>
```

The `depth=1` tells git to only pull down one commit worth of historical data.

### Install Dependencies

In order to install the dependencies you must run:

```bash
mvn install
```

### Run

In order to run the application, run the class ExampleApp. 

## Documentation

### Preparing the Dataset
For our model, we will use the limited data available at this https://archive.ics.uci.edu/ml/datasets/iris

The data is in the form of CSV file. We have total 150 records. Out of that, we will choose random 3 for our testing and rest of the data we will use to train our model. So I have created 2 files for this data. iris.csv and iris-test.csv. The data in a file is like:

```
5.9,3.0,5.1,1.8,2
5.8,2.6,4.0,1.2,1
4.4,3.2,1.3,0.2,0
```

The last column is a classifier and the classification is:

```
0 = "Iris-setosa"
1 = "Iris-versicolor"
2 = "Iris-virginica"
```

### Reading the Data
Deeplearning4j used DataVec libraries to read the data from the different sources and convert them to machine-readable format i.e. Numbers. These numbers are called Vectors and process is called Vectorization.

In our example we are dealing with CSV, so we will use CSVRecordReader. We will put together a simple utility function that accepts file path, batch size, label index and a number of classes.

```
public DataSet readCSVDataset(String csvFileClasspath, int batchSize, int labelIndex, int numClasses)
		throws IOException, InterruptedException {
 
	RecordReader rr = new CSVRecordReader();
	rr.initialize(new FileSplit(new ClassPathResource(csvFileClasspath).getFile()));
	DataSetIterator iterator = new RecordReaderDataSetIterator(rr, batchSize, labelIndex, numClasses);
	return iterator.next();
}
```

Above function will be used to read training as well as test data.


```
int labelIndex = 4;
int numClasses = 3;
 
int batchSizeTraining = 147;
DataSet trainingData = readCSVDataset(irisDataTrainFile, batchSizeTraining, labelIndex, numClasses);
 
// shuffle our training data to avoid any impact of ordering
trainingData.shuffle();
 
int batchSizeTest = 3;
DataSet testData = readCSVDataset(irisDataTestFile, batchSizeTest, labelIndex, numClasses);`
```

The training data we have for iris flowers are labeled that means someone took the pain and classified the training data into 3 different classes. So we need to tell our CSV reader what column out of all data is for a label. Also, we need to specify how many total possible classes are there along with batch size to read.

Note above that I have shuffled the training data so that our model won’t get affected by the ordering of the data. We will not shuffle of test data as we need to refer them later in the example.

For our demonstration and testing purpose, we need to convert this test CSV data to an Object. We will define a simple class to map all those columns in test data:


```
public class Iris {
 
	private Double sepalLength;
	private Double sepalWidth;
	private Double petalLength;
	private Double petalWidth;
 
	private String irisClass;
	
	//Getters; Setters
}
```

Now let’s write a simple utility method to get our objects.

```
private Map<Integer, Iris> objectify(DataSet testData) {
	Map<Integer, Iris> iFlowers = new HashMap<>();
	INDArray features = testData.getFeatureMatrix();
	for (int i = 0; i < features.rows(); i++) {
		INDArray slice = features.slice(i);
		Iris irs = new Iris(slice.getDouble(0), slice.getDouble(1), slice.getDouble(2), slice.getDouble(3));
		iFlowers.put(i, irs);
	}
	return iFlowers;
}
```

This will read and create an object from your dataset:


```
Map<Integer, Iris> flowers = objectify(testData);
flowers.forEach((k, v) -> System.out.println("Index:" + k + " -> " + v));
```


```
Index:0 -> Iris Class = null, Data[ Sepal Length = 5.9, Sepal Width = 3.0, Petal Length = 5.1, Petal Width = 1.8 ]
Index:1 -> Iris Class = null, Data[ Sepal Length = 5.8, Sepal Width = 2.6, Petal Length = 4.0, Petal Width = 1.2 ]
Index:2 -> Iris Class = null, Data[ Sepal Length = 4.4, Sepal Width = 3.2, Petal Length = 1.3, Petal Width = 0.2 ]
```

### Normalizing the Data
In Deep Learning, all the data has to be converted to a specific range. either 0,1 or -1,1. This process is called normalization.


```
// Neural nets all about numbers. Lets normalize our data
DataNormalization normalizer = new NormalizerStandardize();
// Collect the statistics from the training data. This does
// not modify the input data
normalizer.fit(trainingData);
 
// Apply normalization to the training data
normalizer.transform(trainingData);
 
// Apply normalization to the test data.
normalizer.transform(testData);
```

### Building the Network Model

```
int numInputs = 4;
int outputNum = 3;
int iterations = 3000;
long seed = 123;
 
MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
		.seed(seed)
		.iterations(iterations)
		.activation(Activation.TANH)
		.weightInit(WeightInit.XAVIER)
		.learningRate(0.01)
		.regularization(true).l2(1e-4)
		.list()
		.layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3).build())
		.layer(1, new DenseLayer.Builder().nIn(3).nOut(3).build())
		.layer(2,
				new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.activation(Activation.SOFTMAX).nIn(3).nOut(outputNum).build())
		.backprop(true)
		.pretrain(false)
	.build();
```
Deeplearning4j provides an elegant way to build your neural networks with NeuralNetConfiguration. Though it looks simple, a lot of happening in the background and there are many parameters to try and build your model with.

iterations() – This specifies the number of optimization iterations performing multiple passes on the training set.

activation() – It is a function that runs inside a node to determine its output. DL4j supports many such activation functions

weightInit() –  This method specifies one of the many ways to set up the initial weights for the network.

learningRate() – This is one of the crucial parameters to set. This will decide how your model learns to go near desired results. You need to try many parameters before you arrive at almost correct results.

regularization() – Sometimes during training the model, it runs into overfitting and produces bad results for actual data. So we have to regularize it and penalize it if it overfits.

### Train the Network Model

```
MultiLayerNetwork model = new MultiLayerNetwork(conf);
model.init();
model.setListeners(new ScoreIterationListener(100));
 
model.fit(trainingData);
```

Training model is as easy as calling the fit method on your model. You can also set listeners to log the scores.

### Evaluating the Model

```
// evaluate the model on the test set
Evaluation eval = new Evaluation(3);
INDArray output = model.output(testData.getFeatureMatrix());
 
eval.eval(testData.getLabels(), output);
 
System.out.println(eval.stats());
```

For evaluation, you need to provide possible classes that can be one of the outcomes. You get the features (data excluding the labels) from your test data and pass that through your model. When you print stats you will get something like:

```
Examples labeled as 0 classified by model as 0: 1 times
Examples labeled as 1 classified by model as 1: 1 times
Examples labeled as 2 classified by model as 2: 1 times
 
==========================Scores========================================
 Num of classes:    3
 Accuracy:        1.0000
 Precision:       1.0000
 Recall:          1.0000
 F1 Score:        1.0000
Precision, recall & F1: macro-averaged (equally weighted avg. of 3 classes)
========================================================================
```

The model does not predict the actual class for you. It only assigns high values to a class which it think is more correct. If I print the output it will be like

```
[[0.00,  0.12,  0.87],  
 [0.01,  0.96,  0.03],  
 [0.98,  0.02,  0.00]]
```

So what does it tell? we have mentioned that there could be 3 classes for each of the results. So our model has given us INDArray with some values assigned to in each of the index (0,1,2). These indices correspond to classes we defined earlier.

### Classify the Results
We will write a simple utility to get the index of maximum value for a particular row. With this index, we will fetch our actual class name. Remember we did not shuffle the test data. That is because we need to map each of the test data to its output prediction.

```
private void classify(INDArray output, Map<Integer, Iris> flowers) {
	for (int i = 0; i < output.rows(); i++) {
		Iris irs = flowers.get(i);
		// set the classification from the fitted results
		irs.setIrisClass(classifiers.get(maxIndex(getFloatArrayFromSlice(output.slice(i)))));
	}
}
 
private float[] getFloatArrayFromSlice(INDArray rowSlice) {
	float[] result = new float[rowSlice.columns()];
	for (int i = 0; i < rowSlice.columns(); i++) {
		result[i] = rowSlice.getFloat(i);
	}
	return result;
}
 
private static int maxIndex(float[] vals) {
	int maxIndex = 0;
	for (int i = 1; i < vals.length; i++) {
		float newnumber = vals[i];
		if ((newnumber > vals[maxIndex])) {
			maxIndex = i;
		}
	}
	return maxIndex;
}
```

The above utility methods will populate the Iris object with predicted class. Below is the output for that.


```
Index:0 -> Iris Class = Iris-virginica, Data[ Sepal Length = 5.9, Sepal Width = 3.0, Petal Length = 5.1, Petal Width = 1.8 ]
Index:1 -> Iris Class = Iris-versicolor, Data[ Sepal Length = 5.8, Sepal Width = 2.6, Petal Length = 4.0, Petal Width = 1.2 ]
Index:2 -> Iris Class = Iris-setosa, Data[ Sepal Length = 4.4, Sepal Width = 3.2, Petal Length = 1.3, Petal Width = 0.2 ]
```

[opencodez]: https://www.opencodez.com/java/deeplearaning4j.htm
[dl4j]: https://deeplearning4j.org/index.html
[git]: https://git-scm.com/
[maven]: https://maven.apache.org/download.cgi
[jdk-download]: http://www.oracle.com/technetwork/java/javase/downloads

