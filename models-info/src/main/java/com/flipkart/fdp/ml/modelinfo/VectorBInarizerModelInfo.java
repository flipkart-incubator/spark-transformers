package com.flipkart.fdp.ml.modelinfo;



import com.flipkart.fdp.ml.transformer.Transformer;
import com.flipkart.fdp.ml.transformer.VectorBinarizerTranformer;
import lombok.Data;

/**
 * Represents information for a Vector Binarizer model
 * Created by karan.verma on 09/11/16.
 */
@Data
public class VectorBinarizerModelInfo extends AbstractModelInfo {

    private double threshold;

    /**
     * @return an corresponding {@link com.flipkart.fdp.ml.transformer.VectorBinarizerTranformer} for this model info
     */
    @Override
    public Transformer getTransformer() {
        return new VectorBinarizerTranformer(this);
    }

}
