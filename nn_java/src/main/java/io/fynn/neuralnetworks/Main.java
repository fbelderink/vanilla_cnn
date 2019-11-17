package io.fynn.neuralnetworks;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

import io.fynn.neuralnetworks.layers.*;
import io.fynn.neuralnetworks.numpy.*;

public class Main {

	public static numpy np = new numpy();

	public static void main(String[] args) throws Exception {
		NeuralNetwork nn = new NeuralNetwork();

		nn.add(new Flatten());
		nn.add(new Dense(128, 0.001f,"sigmoid"));
		nn.add(new Dense(10, 0.001f,"sigmoid"));

		Convolution conv = new Convolution(5, new tuple<Integer,Integer>(2, 2),new String[]{"padding","same"});
		narray a = np.normal(0, 1, 3,3,3);
		System.out.println(Arrays.toString(a.get(0)));
		System.out.println(Arrays.toString(conv.feedforward(a).get(0)));

		String path_datasets = "D:/Dev Projects/Machine Learning/NeuralNetworks/Datasets/mnist_dataset_csv/";

		narray[] datasets = loadMNIST(path_datasets + "mnist_train.csv");
		narray x_train = datasets[0];
		narray y_train = datasets[1];

		x_train = x_train.divide(255.0f).multiply(0.99f).add(0.01f);

		nn.train(x_train, y_train, 1, "mse");

		narray[] test_datasets = loadMNIST(path_datasets + "mnist_test.csv");
		narray x_test = test_datasets[0];
		narray y_test = test_datasets[1];

		x_test = x_test.divide(255.0f).multiply(0.99f).add(0.01f);

		ArrayList<Float> scorecard = new ArrayList<Float>();

		for(int t = 0; t < x_test.shape(0); t++){
			narray input = new narray(x_test.get(t), Arrays.copyOfRange(x_test.shape(), 1, x_test.shape().length));
			if(np.argmax(nn.predict(input)) ==(int) (y_test.get(t)[0])){
				scorecard.add(1.0f);
			}else{
				scorecard.add(0.0f);
			}
		}

		float sum = 0;
		for(int i = 0; i< scorecard.size(); i++){
			sum += scorecard.get(i);
		}

		System.out.println("performance is: " + (sum / scorecard.size() * 100) + "%");
	}

	public static narray[] loadMNIST(String path) throws IOException {
		narray[] output = new narray[2];

		ArrayList<Float[]> x_train = new ArrayList<Float[]>();
		ArrayList<Integer> y_train = new ArrayList<Integer>();

		BufferedReader reader = new BufferedReader(new FileReader(path));
		String row = null;
		while( ( row = reader.readLine()) != null){
			String[] data = row.split(",");

			ArrayList<Float> x = new ArrayList<>();
			for(int i = 1; i < data.length; i++){
				x.add(Float.parseFloat(data[i]));
			}

			Float[] x_arr = x.toArray(new Float[0]);

			x_train.add(x_arr);
			y_train.add(Integer.parseInt(data[0]));
		}
		reader.close();

		float[] x_train_arr = new float[x_train.size() * 784];
		for(int i = 0; i< x_train.size(); i++){
			for(int j = 0; j < x_train.get(i).length; j++){ 
				x_train_arr[i * x_train.get(i).length +j] = x_train.get(i)[j];
			}
		}

		float[] y_train_arr = new float[y_train.size()];
		for(int i = 0; i < y_train_arr.length; i++){
			y_train_arr[i] = y_train.get(i);
		}

		output[0] = new narray(x_train_arr, x_train.size(),28,28);
		output[1] = new narray(y_train_arr, y_train.size());

		return output;
	}
}