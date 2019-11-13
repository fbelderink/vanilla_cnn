package io.fynn.neuralnetworks;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import io.fynn.neuralnetworks.layers.Layer;
import io.fynn.neuralnetworks.layers.loss.*;
import io.fynn.neuralnetworks.numpy.narray;
import io.fynn.neuralnetworks.numpy.numpy;

public class NeuralNetwork{

	ArrayList<Layer> layers;
	numpy np = new numpy();
	Loss derivative_errors;

	public NeuralNetwork(){
		layers =  new ArrayList<Layer>();
	}

	public narray predict(narray input) throws Exception{
		for(int i = 0; i < layers.size(); i++){
			input = layers.get(i).feedforward(input);
		}
		
		//System.out.println(this.layers.get(this.layers.size() - 1).getOutput().asString());

		return this.layers.get(this.layers.size() - 1).getOutput();
	}

	public void train(narray x_train,narray y_train,int epochs,String loss) throws Exception{
		ArrayList<Float> scorecard = new ArrayList<Float>();
		this.derivative_errors = new Loss(loss);

		for(int e = 0; e < epochs; e++){
			System.out.println("epoch " + (e + 1));
			for(int t = 0; t < x_train.shape(0); t++){

				narray input = new narray(x_train.get(t), Arrays.copyOfRange(x_train.shape(), 1, x_train.shape().length));

				this.predict(input);

				Collections.reverse(this.layers);

				narray targets = np.zeros(10,1);
				targets.set(1,(int) (y_train.get(t)[0]),0);

				if(np.argmax(this.layers.get(0).getOutput()) == (int) (y_train.get(t)[0])){
					scorecard.add(1.0f);
				}else{
					scorecard.add(0.0f);
				}

				if(t % 1000 == 0 && t != 0){
					float sum = 0.0f;
					for(int i = 0; i < scorecard.size(); i++){
						sum += scorecard.get(i);
					}

					System.out.println(t + "/" + y_train.length() + " accuracy:" + (sum / 1000 * 100) + "%\n");
					scorecard.clear();
				}

				narray error =  this.derivative_errors.process(targets, this.layers.get(0).getOutput());

				for(int i = 0; i < this.layers.size(); i++){
					error = this.layers.get(i).backprop(error, loss, i);
				}

				Collections.reverse(this.layers);
			}
		}
	}

	public void add(Layer layer){
		layers.add(layer);
	}
}