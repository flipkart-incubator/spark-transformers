package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.OneHotEncoderModelInfo;
import org.apache.spark.ml.attribute.Attribute;
import org.apache.spark.ml.attribute.AttributeType;
import org.apache.spark.ml.attribute.BinaryAttribute;
import org.apache.spark.ml.attribute.NominalAttribute;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.sql.DataFrame;

/**
 * Transforms Spark's {@link StringIndexerModel} in MlLib to  {@link com.flipkart.fdp.ml.modelinfo.StringIndexerModelInfo} object
 * that can be exported through {@link com.flipkart.fdp.ml.export.ModelExporter}
 */
public class OneHotEncoderModelInfoAdapter implements ModelInfoAdapter<OneHotEncoder, OneHotEncoderModelInfo> {

    @Override
    public OneHotEncoderModelInfo getModelInfo(final OneHotEncoder from, DataFrame df) {
        OneHotEncoderModelInfo modelInfo = new OneHotEncoderModelInfo();
        String inputColumn = from.getInputCol();

        //Ugly but the only way to deal with spark here
        int numTypes = -1;
        Attribute attribute = Attribute.fromStructField(df.schema().apply(inputColumn));
        if(attribute.attrType() == AttributeType.Nominal()) {
            numTypes = ((NominalAttribute) Attribute.fromStructField(df.schema().apply(inputColumn))).values().get().length;
        }else if(attribute.attrType() == AttributeType.Binary()){
            numTypes = ((BinaryAttribute) Attribute.fromStructField(df.schema().apply(inputColumn))).values().get().length;
        }
        //TODO: find a way to extract this from spark OHE instance
        boolean shouldDropLast = true;

        modelInfo.setShouldDropLast(shouldDropLast);
        modelInfo.setNumTypes(numTypes);
        return modelInfo;
    }

    @Override
    public Class<OneHotEncoder> getSource() {
        return OneHotEncoder.class;
    }

    @Override
    public Class<OneHotEncoderModelInfo> getTarget() {
        return OneHotEncoderModelInfo.class;
    }
}
