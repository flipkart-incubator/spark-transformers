package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.CountVectorizerModelInfo;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Transforms input/ predicts for a Count vectorizer model representation
 * captured by  {@link com.flipkart.fdp.ml.modelinfo.CountVectorizerModelInfo}.
 */
public class CountVectorizerTransformer extends TransformerBase {
    private final CountVectorizerModelInfo modelInfo;
    private final Map<String, Integer> vocabulary;

    public CountVectorizerTransformer(final CountVectorizerModelInfo modelInfo) {
        this.modelInfo = modelInfo;
        vocabulary = new HashMap<String, Integer>();
        for( int i =0 ; i < modelInfo.getVocabulary().length; i++) {
            vocabulary.put(modelInfo.getVocabulary()[i], i);
        }
    }

    double[] predict(final String[] input) {
        final Map<String, Integer> termFrequencies = new HashMap<String, Integer>();
        final int tokenCount = input.length;
        for(String term : input) {
            if(vocabulary.containsKey(term)) {
                if(termFrequencies.containsKey(term)) {
                    termFrequencies.put(term, termFrequencies.get(term)+1);
                }
                else {
                    termFrequencies.put(term, 1);
                }
            }
            else{
                //ignore terms not in vocabulary
            }
        }
        final int effectiveMinTF = (int)( (modelInfo.getMinTF() >= 1.0) ? modelInfo.getMinTF() : modelInfo.getMinTF() * tokenCount);

        final double[] encoding = new double[modelInfo.getVocabSize()];
        Arrays.fill(encoding, 0.0);

        for(final Map.Entry<String, Integer> entry : termFrequencies.entrySet()) {
            //filter out terms with freq < effectiveMinTF
            if( entry.getValue() >= effectiveMinTF) {
                int position = vocabulary.get(entry.getKey());
                encoding[position] = entry.getValue();
            }
        }
        return encoding;
    }

    @Override
    public void transform(Map<String, Object> input) {
        String[] inp = (String []) input.get(getInputKeys()[0]);
        input.put(getOutputKey(), predict(inp));
    }
}
