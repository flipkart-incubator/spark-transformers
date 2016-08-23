package com.flipkart.fdp.ml.utils;

import com.flipkart.fdp.ml.transformer.Transformer;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class PipelineUtilsTest {

    @Test
    public void testExtractedInputForSingleTransformer() {
        Transformer t = createTransformer(Arrays.asList("a"), Arrays.asList("b"));
        Set<String> inputs = PipelineUtils.extractRequiredInputColumns(new Transformer[] {t});
        assertEquals(inputs.size(), 1);
        assertTrue(inputs.containsAll(Arrays.asList("a")));
        assertFalse(inputs.contains("b"));
    }

    @Test
    public void testExtractedInputForSingleTransformerWithColumnModification() {
        Transformer t = createTransformer(Arrays.asList("a", "b", "c"), Arrays.asList("a1", "b1", "c"));
        Set<String> inputs = PipelineUtils.extractRequiredInputColumns(new Transformer[] {t});
        assertEquals(inputs.size(), 3);
        assertTrue(inputs.containsAll(Arrays.asList("a","b","c")));
        assertFalse(inputs.contains("a1"));
        assertFalse(inputs.contains("b1"));
    }

    @Test
    public void testExtractedInputForIndependentTransformers() {
        Transformer t1 = createTransformer(Arrays.asList("a", "b", "c"), Arrays.asList("a1", "b1", "c"));
        Transformer t2 = createTransformer(Arrays.asList("d", "e", "f"), Arrays.asList("d1", "e1", "f1"));
        Set<String> inputs = PipelineUtils.extractRequiredInputColumns(new Transformer[] {t1, t2});
        assertEquals(inputs.size(), 6);
        assertTrue(inputs.containsAll(Arrays.asList("a","b","c","d","e","f")));
        assertFalse(inputs.contains("a1"));
        assertFalse(inputs.contains("b1"));
        assertFalse(inputs.contains("d1"));
        assertFalse(inputs.contains("e1"));
        assertFalse(inputs.contains("f1"));
    }

    @Test
    public void testExtractedInputForDependentTransformers() {
        Transformer t1 = createTransformer(Arrays.asList("a", "b", "c"), Arrays.asList("a1", "b1", "c"));
        Transformer t2 = createTransformer(Arrays.asList("a1", "c", "f"), Arrays.asList("a2", "c1"));
        Transformer t3 = createTransformer(Arrays.asList("a", "a1", "a2"), Arrays.asList("a3", "a4"));
        Set<String> inputs = PipelineUtils.extractRequiredInputColumns(new Transformer[] {t1, t2, t3});
        assertEquals(inputs.size(), 4);
        assertTrue(inputs.containsAll(Arrays.asList("a","b","c","f")));
        assertFalse(inputs.contains("a1"));
        assertFalse(inputs.contains("b1"));
        assertFalse(inputs.contains("a2"));
        assertFalse(inputs.contains("c1"));
        assertFalse(inputs.contains("a3"));
        assertFalse(inputs.contains("a4"));
    }

    @Test
    public void testExtractedInputForTransformersWithModification() {
        Transformer t1 = createTransformer(Arrays.asList("a", "b", "c"), Arrays.asList("a1", "b1", "c"));
        Transformer t2 = createTransformer(Arrays.asList("a1", "c", "f"), Arrays.asList("a2", "c1"));
        Transformer t3 = createTransformer(Arrays.asList("a", "a1", "a2"), Arrays.asList("a2"));
        Set<String> inputs = PipelineUtils.extractRequiredInputColumns(new Transformer[] {t1, t2, t3});
        assertEquals(inputs.size(), 4);
        assertTrue(inputs.containsAll(Arrays.asList("a","b","c","f")));
        assertFalse(inputs.contains("a1"));
        assertFalse(inputs.contains("b1"));
        assertFalse(inputs.contains("a2"));
        assertFalse(inputs.contains("c1"));
    }

    @Test
    public void testExtractedOutputForSingleTransformer() {
        Transformer t = createTransformer(Arrays.asList("a"), Arrays.asList("b"));
        Set<String> outputs = PipelineUtils.extractRequiredOutputColumns(new Transformer[] {t});
        assertEquals(outputs.size(), 1);
        assertTrue(outputs.containsAll(Arrays.asList("b")));
        assertFalse(outputs.contains("a"));
    }

    @Test
    public void testExtractedOutputForSingleTransformerWithColumnModification() {
        Transformer t = createTransformer(Arrays.asList("a", "b", "c"), Arrays.asList("a1", "b1", "c"));
        Set<String> outputs = PipelineUtils.extractRequiredOutputColumns(new Transformer[] {t});
        assertEquals(outputs.size(), 3);
        assertTrue(outputs.containsAll(Arrays.asList("a1","b1","c")));
        assertFalse(outputs.contains("a"));
        assertFalse(outputs.contains("b"));
    }

    @Test
    public void testExtractedOutputForIndependentTransformers() {
        Transformer t1 = createTransformer(Arrays.asList("a", "b", "c"), Arrays.asList("a1", "b1", "c"));
        Transformer t2 = createTransformer(Arrays.asList("d", "e", "f"), Arrays.asList("d1", "e1", "f1"));
        Set<String> outputs = PipelineUtils.extractRequiredOutputColumns(new Transformer[] {t1, t2});
        assertEquals(outputs.size(), 6);
        assertTrue(outputs.containsAll(Arrays.asList("a1","b1","c","d1","e1","f1")));
        assertFalse(outputs.contains("a"));
        assertFalse(outputs.contains("b"));
        assertFalse(outputs.contains("d"));
        assertFalse(outputs.contains("e"));
        assertFalse(outputs.contains("f"));
    }

    @Test
    public void testExtractedOutputForDependentTransformers() {
        Transformer t1 = createTransformer(Arrays.asList("a", "b", "c"), Arrays.asList("a1", "b1", "c"));
        Transformer t2 = createTransformer(Arrays.asList("a1", "c", "f"), Arrays.asList("a2", "c1"));
        Transformer t3 = createTransformer(Arrays.asList("a", "a1", "a2"), Arrays.asList("a3", "a4"));
        Set<String> outputs = PipelineUtils.extractRequiredOutputColumns(new Transformer[] {t1, t2, t3});
        assertEquals(outputs.size(), 4);
        assertTrue(outputs.containsAll(Arrays.asList("b1","c1","a3","a4")));
        assertFalse(outputs.contains("a"));
        assertFalse(outputs.contains("b"));
        assertFalse(outputs.contains("a1"));
        assertFalse(outputs.contains("c"));
        assertFalse(outputs.contains("f"));
        assertFalse(outputs.contains("a1"));
        assertFalse(outputs.contains("a2"));
    }

    @Test
    public void testExtractedOutputForTransformersWithModification() {
        Transformer t1 = createTransformer(Arrays.asList("a", "b", "c"), Arrays.asList("a1", "b1", "c"));
        Transformer t2 = createTransformer(Arrays.asList("a1", "c", "f"), Arrays.asList("a2", "c1"));
        Transformer t3 = createTransformer(Arrays.asList("a", "a1", "a2"), Arrays.asList("a2"));
        Set<String> outputs = PipelineUtils.extractRequiredOutputColumns(new Transformer[] {t1, t2, t3});
        assertEquals(outputs.size(), 4);
        assertTrue(outputs.containsAll(Arrays.asList("b1","a2","c1","f")));
        assertFalse(outputs.contains("a"));
        assertFalse(outputs.contains("b"));
        assertFalse(outputs.contains("c"));
        assertFalse(outputs.contains("a1"));
    }

    private Transformer createTransformer(final List<String> inputs, final List<String> outputs) {
        return new Transformer() {
            @Override
            public void transform(Map<String, Object> input) {
            }

            @Override
            public Set<String> getInputKeys() {
                return new HashSet<>(inputs);
            }

            @Override
            public Set<String> getOutputKeys() {
                return new HashSet<>(outputs);
            }
        };
    }
}
