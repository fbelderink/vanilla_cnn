package io.fynn.neuralnetworks.numpy;

import java.util.Arrays;

public class numpy{

    public narray zeros(int... shape){
        narray output = new narray(shape);

        output.setAll(0.0f);

        return output;
    }

    public narray normal(float center,float scale,int... shape){
        narray output = zeros(shape);
        for(int i = 0; i < output.length(); i++){
            output.array[i] = randomNumber(-scale, scale, center);
        }
        return output;
    }

    public narray dot(narray X,narray Y) throws Exception{
        if(X.shape[1] != Y.shape[0]){
            throw new Exception("Can't compute shapes " + Arrays.toString(X.shape) + " and " + Arrays.toString(Y.shape) + "!");
        }

        narray output = zeros(X.shape[0], Y.shape[1]);

        for(int i = 0; i < X.shape[0]; i++){
            for(int j = 0; j < Y.shape[1]; j++ ){
                for(int k = 0; k < X.shape[1]; k++){
                    output.set(X.get(i,k)[0] * Y.get(k,j)[0], i,j);
                }
            } 
        }

        return output;
    }

    public float max(narray X){
        float output = 0.0f;

        for(int i = 0; i < X.length(); i++){
            if(output < X.array[i]){
                output = X.array[i];
            }
        }

        return output;
    }
    
    public int argmax(narray X){
        float max = 0.0f;
        int argmax = 0;
        for(int i = 0; i < X.length(); i++){
            if(max < X.getArray()[i]){
                max = X.getArray()[i];
                argmax = i;
            }
        }

        return argmax;
    }

    public float sum(narray X){

        float output = 0.0f;

        for(int i = 0; i < X.length(); i++){
                output += X.array[i];
        }

        return output;
    }

    public narray transpose2D(narray X) throws Exception{
        narray output = new narray(X.shape[1],X.shape[0]);

        for(int i = 0; i < X.shape[0]; i++){
            for(int j = 0; j < X.shape[1]; j++){
                output.set(X.get(i,j)[0], j,i);
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