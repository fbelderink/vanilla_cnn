package io.fynn.neuralnetworks.numpy;

public class numpy{

    public float[][][][] zeros4D(int a,int b, int c, int d){
        float[][][][] output = new float[a][b][c][d];

        for(int i = 0; i < a; i++){
            for( int j = 0 ;j < b; j ++){
                for(int k = 0;k < c; k++ ){
                    for(int l = 0; l < d; l++){
                        output[i][j][k][l] = 0.0f;
                    }
                }
            }
        }
        return output;
    }

    public float[][] zeros2D(int x, int y){
        float[][] output = new float[x][y];
        for(int i = 0; i< x; i++){
            for(int j = 0; j < y; j++){
                output[i][j] = 0.0f;
            }
        }

        return output;
    }

    public float[][] normal(float center,float scale,int x, int y ){
        float[][] output = zeros2D(x, y);
        for(int i = 0; i< x; i++){
            for(int j = 0; j < y; j++){
                output[i][j] = randomNumber(-scale, scale, center);
            }
        }
        return output;
    }

    public float[][] dot(float[][] X,float[][] Y){
        if(X[0].length != Y.length){
            return null;
        }

        float[][] output = zeros2D(X.length, Y[0].length);

        for(int i = 0; i < X.length; i++){
            for(int j = 0; j < Y[0].length; j++ ){
                for(int k = 0; k < X[0].length; k++){
                    output[i][j] = X[i][k] * Y[k][j];
                }
            } 
        }

        return output;
    }

    public float max(float[][][][] X){
        float output = 0.0f;

        for(int i = 0; i < X[0].length; i++){
            for(int j = 0; j < X[1].length; j++){
                for(int k = 0; k < X[2].length; k++){
                    for(int l = 0; l < X[3].length; l++){
                        if(X[i][j][k][l] > output){
                            output = X[i][j][k][l];
                        }
                    }
                }
            }  
        }

        return output;
    }

    public float max(float[][] X){
        float output = 0.0f;

        for(int i = 0; i < X[0].length; i++){
            for(int j = 0; j < X[1].length; j++){
                if(X[i][j] > output){
                    output = X[i][j];
                }
            }
        }

        return output;
    }

    public float sum(float[][] X){

        float output = 0.0f;

        for(int i = 0; i < X[0].length; i++){
            for(int j = 0; j < X[1].length; j++){
                output += X[i][j];
            }
        }

        return output;
    }

    public float sum(float[][][][] X){

        float output = 0.0f;

        for(int i = 0; i < X[0].length; i++){
            for(int j = 0; j < X[1].length; j++){
                for(int k = 0; k < X[2].length; k++){
                    for(int l = 0; l < X[3].length; l++){
                        output += X[i][j][k][l];
                    }
                }
            } 
        }

        return output;
    }


    public float randomNumber(float max,float min,float offset){
        max += offset;
        min += offset;
        return (float) (Math.random() * (max - min) + min);

    }

}