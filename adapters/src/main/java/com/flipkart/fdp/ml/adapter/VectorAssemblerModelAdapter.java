package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.VectorAssemblerModelInfo;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.DataFrame;

import java.util.Arrays;
import java.util.LinkedHashSet;

/**
 * Transforms Spark's {@link VectorAssemblerModel} in MlLib to  {@link com.flipkart.fdp.ml.modelinfo.VectorAssemblerModelInfo} object
 * that can be exported through {@link com.flipkart.fdp.ml.export.ModelExporter}

 * Created by rohan.shetty on 28/03/16.
 */
public class VectorAssemblerModelAdapter extends AbstractModelInfoAdapter<VectorAssembler, VectorAssemblerModelInfo> {

    @Override
    VectorAssemblerModelInfo getModelInfo(VectorAssembler from, DataFrame df) {
        VectorAssemblerModelInfo vectorAssemblerModelInfo = new VectorAssemblerModelInfo();
        vectorAssemblerModelInfo.setInputKeys(new LinkedHashSet<>(Arrays.asList(from.getInputCols())));
        vectorAssemblerModelInfo.setOutputKey(from.getOutputCol());
        return vectorAssemblerModelInfo;
    }

    @Override
    public Class<VectorAssembler> getSource() {
        return VectorAssembler.class;
    }

    @Override
    public Class<VectorAssemblerModelInfo> getTarget() {
        return VectorAssemblerModelInfo.class;
    }
}
