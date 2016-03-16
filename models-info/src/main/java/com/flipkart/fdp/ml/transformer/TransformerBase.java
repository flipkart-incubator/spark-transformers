package com.flipkart.fdp.ml.transformer;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public abstract class TransformerBase implements Transformer {
    private String inputKey = "input", outputKey="output";
}
