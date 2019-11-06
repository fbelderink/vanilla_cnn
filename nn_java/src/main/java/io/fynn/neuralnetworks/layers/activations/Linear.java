package io.fynn.neuralnetworks.layers.activations;

public class Linear{

    public float[][][][] feedfoward(float[][][][] X){
        return X;
    }

    public float[][] feedfoward(float[][] X){
        return X;
    }

    public float[][] backprop(float[][] X){
        
        for(int i = 0; i < X[0].length; i++){
            for(int j = 0; j < X[1].length; j++){
                X[i][j] = 1.0f;
            }
        }
        	
        return X;
    }

    public float[][][][] backprop(float[][][][] X){
        
        for(int i = 0; i < X[0].length; i++){
            for(int j = 0; j < X[1].length; j++){
                for(int k = 0; k < X[2].length; k++){
                    for(int l = 0; l < X[3].length; l++){
                        X[i][j][k][l] = 1.0f;
                    }
                }
            }
        }

        return X;
    }

}