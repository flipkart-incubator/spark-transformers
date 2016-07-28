package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.AbstractModelInfo;
import com.flipkart.fdp.ml.modelinfo.RegexTokenizerModelInfo;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Transforms input/ predicts for a Regex Tokenizer model representation
 * captured by  {@link com.flipkart.fdp.ml.modelinfo.RegexTokenizerModelInfo}.
 */
public class RegexTokenizerTransformer implements Transformer {
    private final RegexTokenizerModelInfo modelInfo;

    public RegexTokenizerTransformer(final RegexTokenizerModelInfo modelInfo) {
        this.modelInfo = modelInfo;
    }

    public String[] predict(final String input) {
        final Pattern regex = Pattern.compile(modelInfo.getPattern());
        final String targetStr = (modelInfo.isToLowercase() ? input.toLowerCase() : input);
        final List<String> tokens;
        if (modelInfo.isGaps()) {
            //using linkedlist for efficient deletion while filtering
            tokens = new LinkedList<String>(Arrays.asList(targetStr.split(regex.pattern())));
        } else {
            List<String> allMatches = new LinkedList<>();
            Matcher m = regex.matcher(targetStr);
            while (m.find()) {
                allMatches.add(m.group());
            }
            tokens = allMatches;
        }
        tokens.removeIf(new Predicate<String>() {
            @Override
            public boolean test(String p) {
                return p.length() < modelInfo.getMinTokenLength();
            }
        });
        final String[] filteredTokens = new String[tokens.size()];
        for (int i = 0; i < filteredTokens.length; i++) {
            filteredTokens[i] = tokens.get(i);
        }
        return filteredTokens;
    }

    @Override
    public void transform(Map<String, Object> input) {
        String inp = (String) input.get(modelInfo.getInputKeys().iterator().next());
        input.put(modelInfo.getOutputKey(), predict(inp));
    }
}
