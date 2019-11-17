package io.fynn.neuralnetworks.layers;

import java.util.Arrays;

import io.fynn.neuralnetworks.layers.activations.Activation;
import io.fynn.neuralnetworks.numpy.narray;
import io.fynn.neuralnetworks.numpy.numpy;
import io.fynn.neuralnetworks.numpy.tuple;

public class Convolution extends Layer {
    
    numpy np = new numpy();
    
    int num_filters;
    tuple<Integer,Integer> filter_size,strides;
    String padding,activation;
    float bias,lr;

    narray input,output;
    int[] input_shape;
    narray filters;
    Activation activation_function;

    public Convolution(int num_filters, tuple<Integer,Integer> filter_size, tuple<Integer,Integer> strides,String padding,String activation,float bias,float lr){
        this.num_filters = num_filters;
        this.filter_size = filter_size;
        this.strides = strides;
        this.padding = padding;
        this.activation = activation;
        this.bias = bias;
        this.lr = lr;
        this.filters = np.normal(0.0f, 1.0f, num_filters,filter_size.getFirst(),filter_size.getSecond());
        this.activation_function = new Activation(activation);
    }

    public Convolution(int num_filters,tuple<Integer,Integer> filter_size, tuple<Integer,Integer> strides,String[] t) throws Exception{
        if(t.length != 2){
            throw new Exception("String[] has to be of length 2!");
        }

        this.num_filters = num_filters;
        this.filter_size = filter_size;
        this.strides = strides;
        this.bias = 0.1f;
        this.lr = 0.001f;
        this.filters = np.normal(0.0f, 1.0f, num_filters,filter_size.getFirst(),filter_size.getSecond());
        if(t[0].equals("padding")){
            this.padding = t[1];
            this.activation = "linear";
        }else if(t[1].equals("activation")){
            this.activation = t[1];
            this.padding = "valid";
        }else{
            throw new Exception("There is no argument called " + t[0] + "!");
        }
        this.activation_function = new Activation(this.activation);
    }

    public Convolution(int num_filters,tuple<Integer,Integer> filter_size,String[] t) throws Exception{
        if(t.length != 2){
            throw new Exception("String[] has to be of length 2!");
        }

        this.num_filters = num_filters;
        this.filter_size = filter_size;
        this.strides = new tuple<Integer,Integer>(1,1);
        this.bias = 0.1f;
        this.lr = 0.001f;
        this.filters = np.normal(0.0f, 1.0f, num_filters,filter_size.getFirst(),filter_size.getSecond());
        if(t[0].equals("padding")){
            this.padding = t[1];
            this.activation = "linear";
        }else if(t[1].equals("activation")){
            this.activation = t[1];
            this.padding = "valid";
        }else{
            throw new Exception("There is no argument called " + t[0] + "!");
        }
        this.activation_function = new Activation(this.activation);
    }

    public Convolution(int num_filters,tuple<Integer,Integer> filter_size, tuple<Integer,Integer> strides){
        this(num_filters,filter_size,strides,"valid","linear",0.1f,0.001f);
    }

    public Convolution(int num_filters,tuple<Integer,Integer> filter_size){
        this(num_filters,filter_size,new tuple<Integer,Integer>(1,1),"valid","linear",0.1f,0.001f);
    }

    @Override
    public narray feedforward(narray X) throws Exception {
        this.input = X.clone();

        this.input_shape = X.shape();

        System.out.println(Arrays.toString(this.filters.get(0)));

        int image_layers = X.shape(0);

        int image_dim_i = X.shape(1);
        int image_dim_j = X.shape(2);

        int output_dim_i,output_dim_j = 0;

        if(this.padding.equals("same")){
            output_dim_i = image_dim_i;
            output_dim_j = image_dim_j;
        }else if(this.padding.equals("valid")){
            output_dim_i = (image_dim_i - this.filter_size.getFirst()) / this.strides.getFirst() + 1;
            output_dim_j = (image_dim_j - this.filter_size.getSecond()) / this.strides.getSecond() + 1;
        }else{
            throw new Exception("There is no padding type called " + this.padding);
        }

        this.output = np.zeros(this.num_filters,output_dim_i,output_dim_j);

        for(int filter_i = 0; filter_i < this.num_filters; filter_i++){
            narray current_filter = new narray(this.filters.get(filter_i),this.filter_size.getFirst(),this.filter_size.getSecond());
            for(int image_i = 0; image_i < image_layers; image_i++){
                this.output.setadd(new int[]{image_i},this.convolve2d(new narray(X.get(image_i), image_dim_i,image_dim_j),current_filter,new tuple<Integer,Integer>(output_dim_i, output_dim_j)).getArray());
            }
        }

        this.output.add(this.bias);

        this.output = this.activation_function.feedforward(this.output);

        return this.output;
    }

    @Override
    public narray backprop(narray E, String loss, int layer_i) throws Exception {
        
        return null;
    }

    @SuppressWarnings("unchecked")
    public narray convolve2d(narray X,narray filter,tuple<Integer,Integer> output_dim) throws Exception{

        if(this.padding.equals("same")){
            int padding_size_i = ((output_dim.getFirst() - 1) * this.strides.getFirst() + this.filter_size.getFirst()) - output_dim.getFirst();
            int padding_size_j = ((output_dim.getSecond() - 1) * this.strides.getSecond() + this.filter_size.getSecond()) - output_dim.getSecond();

            X = this.zeropad(X,new tuple<Integer,Integer>(padding_size_i, padding_size_j));

            this.input = X.clone();
        }

        narray convolved_image = np.zeros(output_dim.getFirst(),output_dim.getSecond());
        for(int i = 0; i < output_dim.getFirst(); i++){
            for(int j = 0; j < output_dim.getSecond(); j++){
                narray W = X.slice(new tuple<Integer,Integer>(i * this.strides.getFirst(),i * this.strides.getFirst() + this.filter_size.getFirst()),new tuple<Integer,Integer>(j * this.strides.getSecond(),j * this.strides.getSecond() + this.filter_size.getSecond()));
                //System.out.println(np.sum(W.multiply(filter.getArray())));
                convolved_image.set(np.sum(W.multiply(filter.getArray())), i,j);
            }
        }

        return convolved_image;
    }

    @SuppressWarnings("unchecked")
    public narray zeropad(narray X,tuple<Integer,Integer> padding_size) throws Exception{
        narray output = np.zeros(X.shape(0) + padding_size.getFirst(),X.shape(1) + padding_size.getSecond());

        int padding_size_i = padding_size.getFirst() % 2 == 0 ? padding_size.getFirst() / 2 : padding_size.getFirst() % 2;
        int padding_size_j = padding_size.getSecond() % 2 == 0 ? padding_size.getSecond() / 2 : padding_size.getSecond() % 2;

        for(int i = 0; i < X.length(); i++){
            output.setSlice(X.get1D(i), i, new tuple<Integer,Integer>(padding_size_i, padding_size_i + X.shape(0)),new tuple<Integer,Integer>(padding_size_j, padding_size_j + X.shape(1)));
        }

        return output;
    }


    @Override
    public narray getInput() {
        return this.input;
    }

    @Override
    public narray getOutput() {
        return this.output;
    }
    
}