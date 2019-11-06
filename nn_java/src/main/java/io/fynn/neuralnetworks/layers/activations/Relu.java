package io.fynn.neuralnetworks.layers.activations;

public class Relu{
    
    public float[][][][] feedfoward(float[][][][] X){
        for(int i = 0; i < X[0].length; i++){
            for(int j = 0; j < X[1].length; j++){
                for(int k = 0; k < X[2].length; k++){
                    for(int l = 0; l < X[3].length; l++){
                        if(X[i][j][k][l] <= 0){
                            X[i][j][k][l] = 0.0f;
                        }
                    }
                }
            }
        }

        return X;
    }

    public float[][] feedfoward(float[][] X){
        for(int i = 0; i < X[0].length; i++){
            for(int j = 0; j < X[1].length; j++){
                if(X[i][j] <= 0){
                    X[i][j] = 0.0f; 
                }
            }
        }
        return X;
    }

    public float[][] backprop(float[][] X){
        for(int i = 0; i < X[0].length; i++){
            for(int j = 0; j < X[1].length; j++){
                if(X[i][j] <= 0){
                    X[i][j] = 0.0f;
                }else{
                    X[i][j] = 1.0f;
                }
            } 
        }
        return X;
    }

    public float[][][][] backprop(float[][][][] X){
        for(int i = 0; i < X[0].length; i++){
            for(int j = 0; j < X[1].length; j++){
                for(int k = 0; k < X[2].length; k++){
                    for(int l = 0; l < X[3].length; l++){
                        if(X[i][j][k][l] <= 0){
                            X[i][j][k][l] = 0.0f;
                        }else{
                            X[i][j][k][l] = 1.0f;
                        }
                    }
                }
            }
        }
        return X;
    }
}