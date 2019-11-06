package io.fynn.neuralnetworks.layers.activations;

import io.fynn.neuralnetworks.numpy.*;

public class Softmax{

    numpy np = new numpy();

    public float[][][][] feedfoward(float[][][][] X){
        float max = np.max(X);
        for(int i = 0; i < X[0].length; i++){
            for(int j = 0; j < X[1].length; j++){
                for(int k = 0; k < X[2].length; k++){
                    for(int l = 0; l < X[3].length; l++){
                        
                    }
                }
            }   
        }
        return X;
    }

    public float[][] feedfoward(float[][] X){
        return X;
    }

    public float[][][][] backprop(float[][][][] X){
        return X;
    }  

    public float[][] backprop(float[][] X){
        return X;
    }

}