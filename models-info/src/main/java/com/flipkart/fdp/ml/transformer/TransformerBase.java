package com.flipkart.fdp.ml.transformer;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public abstract class TransformerBase implements Transformer {
    private String[] inputKeys = new String [] {"input"};
    private String outputKey="output";
}
