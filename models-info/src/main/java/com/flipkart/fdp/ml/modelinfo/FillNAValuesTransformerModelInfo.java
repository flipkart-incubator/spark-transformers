package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.FillNAValuesTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;
import lombok.Data;

import java.util.HashMap;
import java.util.Map;


@Data
public class FillNAValuesTransformerModelInfo extends AbstractModelInfo {

    //TODO: types are inferred during deserialization. Integers are being inferred as doubles. Verification is needed if it is a problem
    private Map<String, Object> naValuesMap = new HashMap<>();
    /**
     * @return an corresponding {@link FillNAValuesTransformer} for this model info
     */
    @Override
    public Transformer getTransformer() {
        return new FillNAValuesTransformer(this);
    }
}
