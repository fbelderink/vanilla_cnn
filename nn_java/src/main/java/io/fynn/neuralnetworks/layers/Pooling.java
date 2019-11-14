package io.fynn.neuralnetworks.layers;

import io.fynn.neuralnetworks.numpy.narray;
import io.fynn.neuralnetworks.numpy.numpy;

public class Pooling extends Layer{

    numpy np = new numpy();

    public narray input,output;

    String type;
    int[] pooling_size, strides;

    public Pooling(String type,int[] pooling_size, int[] strides){
        this.type = type;
        this.pooling_size = pooling_size;
        this.strides = strides;
    }

    public Pooling(String type){
        this(type,new int[]{2,2},new int[]{2,2});
    }

    public Pooling(){
        this("max",new int[]{2,2},new int[]{2,2});
    }


    public narray feedforward(narray X) throws Exception{
        this.input = X;

        int image_layers = X.shape(0);
        int image_dim_i = X.shape(1);
        int image_dim_j = X.shape(2);

        int output_dim_i = (image_dim_i - this.pooling_size[0]) / this.strides[0] + 1;
        int output_dim_j = (image_dim_j - this.pooling_size[1]) / this.strides[1] + 1;

        narray pooled_features = np.zeros(image_layers,output_dim_i,output_dim_j);
        for(int image_i = 0; image_i < image_layers; image_i++){
            for(int i = 0; i < image_dim_i; i++){
                for(int j = 0; j < image_dim_j; j++){
                    narray W = X.slice(new int[]{image_i,image_i + 1},new int[]{i * this.strides[0],i * this.strides[0] + this.pooling_size[0]},new int[]{j * this.strides[1],j * this.strides[1] + this.pooling_size[1]});
                    if(this.type == "max"){
                        pooled_features.set(np.max(W), image_i,i,j);
                        //todo where for backprop
                    }else if(this.type == "avg"){
                        pooled_features.set(np.sum(W) / (W.shape(0) * W.shape(1))); 
                    }else{
                        throw new Exception("Choose between max and avg");
                    }
                }
            }
        }
        
        this.output = pooled_features;
        return this.output;
    }

    public narray backprop(narray E,String loss,int layer_i){

        return null;
    }

    public narray getInput(){
        return this.input;
    }

    public narray getOutput(){
        return this.output;
    }


}