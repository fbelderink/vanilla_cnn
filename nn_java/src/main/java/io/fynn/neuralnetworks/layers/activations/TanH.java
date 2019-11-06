package io.fynn.neuralnetworks.layers.activations;

public class TanH{

    public float[][][][] feedfoward(float[][][][] X){
        for(int i = 0; i < X[0].length; i++){
            for(int j = 0; j < X[1].length; j++){
                for(int k = 0; k < X[2].length; k++){
                    for(int l = 0; l < X[3].length; l++){
                        X[i][j][k][l] = (float) (2.0f / (1.0f + Math.exp(-2 * X[i][j][k][l])) - 1);
                    }
                }
            }
        }

        return X;
    }

    public float[][] feedfoward(float[][] X){
        for(int i = 0; i < X[0].length; i++){
            for(int j = 0; j < X[0].length; j++){
                X[i][j] = (float) (2.0f / (1.0f + Math.exp(-2 * X[i][j])) - 1);
            }
        }

        return X;
    }

    public float[][][][] backprop(float[][][][] X){
        for(int i = 0; i < X[0].length; i++){
            for(int j = 0; j < X[0].length; j++){
                for(int k = 0; k < X[0].length; k++){
                    for(int l = 0; l < X[0].length; l++){
                        X[i][j][k][l] = (float) (1.0f - Math.pow(X[i][j][k][l],2));
                    }
                }
            }
        }
        return X;
    }

    public float[][] backprop(float[][] X){
        for(int i = 0; i < X[0].length; i++){
            for(int j = 0; j < X[1].length; j++){
                X[i][j] = (float) (1.0f - Math.pow(X[i][j],2));
            }
        }
        return X;
    }

}