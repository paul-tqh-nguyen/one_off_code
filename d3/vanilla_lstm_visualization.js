
const vanillaLSTMVisualizationMain = () => {

    const svg = d3.select('#vanilla-lstm-visualization-svg');
    const wordsGroup = svg.append('g');
    const hashingArrowsGroup = svg.append('g');    

    const words = 'This film is an expert work of new myths.'.split(' ');
    const wordTextFontSize = 15;

    const arrowPadding = 25;
    
    const hashingArrowHeight = 100;
    
    const margin = {
        top: 10,
        bottom: 80,
        left: 80,
        right: 80,
    };
    const borderColor = '#b69cff';
    
    svg.append('defs').append('marker')
        .attr('id', 'triangle')
        .attr('refX', 12)
        .attr('refY', 12)
        .attr('markerWidth', 30)
        .attr('markerHeight', 30)
        .attr('markerUnits','userSpaceOnUse')
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M 0 0 24 12 0 24 6 12')
        .style('fill', borderColor);
    
    const redraw = () => {
        
        const plotcontainer = document.getElementById('vanilla-lstm-visualization');
        svg
            .attr('width', plotcontainer.clientWidth)
            .attr('height', plotcontainer.clientHeight);
        
        const svgWidth = parseFloat(svg.attr('width'));
        const svgHeight = parseFloat(svg.attr('height'));
        
        const innerWidth = svgWidth - margin.left - margin.right;
        const innerHeight = svgHeight - margin.top - margin.bottom;
        
        const wordTextOuterBoxWidth = innerWidth / words.length;
        const centerXForIndex = i => i * wordTextOuterBoxWidth + margin.left + wordTextOuterBoxWidth / 2;
        const wordRectWidth = wordTextOuterBoxWidth * 0.8;
        wordsGroup.selectAll('*').remove();
        hashingArrowsGroup.selectAll('*').remove();
        words.forEach((word, wordIndex) => {
            const wordRectHeight = wordTextFontSize * 2;
            wordsGroup
                .append('rect')
                .style('stroke-width', 3)
                .style('stroke', borderColor)
                .attr('x', centerXForIndex(wordIndex) - wordTextOuterBoxWidth / 2)
                .attr('y', margin.top)
                .attr('width', wordRectWidth)
                .attr('height', wordRectHeight)
                .attr('fill', '#fff');
            
            const wordId = `word-${wordIndex}`;
            wordsGroup
                .append('text')
                .html(word)
                .style('text-anchor', 'middle')
                .style('font-size', `${wordTextFontSize}px`)
                .style('font-family', 'Libre Baskerville')
                .attr('id', wordId);
            const wordDOMElement = d3.select('#'+wordId);
            const wordDOMElementBBox = wordDOMElement.node().getBBox();
            wordDOMElement
                .attr('x', centerXForIndex(wordIndex) + wordDOMElementBBox.width / 2 - wordRectWidth / 2)
                .attr('y', margin.top + wordDOMElementBBox.height * 1.1);

            const hashingArrowStartY = margin.top + wordRectHeight + arrowPadding;
            const hashingArrowEndY = margin.top + wordTextFontSize + hashingArrowHeight;
            hashingArrowsGroup.append('line')
                .attr('x1', centerXForIndex(wordIndex) - wordTextFontSize / 2)
                .attr('y1', hashingArrowStartY)
                .attr('x2', centerXForIndex(wordIndex) - wordTextFontSize / 2)
                .attr('y2', hashingArrowEndY)
                .attr('stroke-width', 4)
                .attr('stroke', borderColor)
                .attr('marker-end', 'url(#triangle)');

            const wordHashRectY = hashingArrowEndY + wordTextFontSize * 2;
            const wordHashRectHeight = wordTextFontSize * 2;
            wordsGroup
                .append('rect')
                .style('stroke-width', 3)
                .style('stroke', borderColor)
                .attr('x', centerXForIndex(wordIndex) - wordTextOuterBoxWidth / 2)
                .attr('y', wordHashRectY)
                .attr('width', wordRectWidth)
                .attr('height', wordRectHeight)
                .attr('fill', '#fff');
            const wordHashId = `word-hash-${wordIndex}`;
            wordsGroup
                .append('text')
                .html(Math.ceil(Math.random()*1234*(1+wordIndex*wordIndex)) % 87 + 10)
                .style('text-anchor', 'middle')
                .style('font-size', `${wordTextFontSize}px`)
                .style('font-family', 'Libre Baskerville')
                .attr('id', wordHashId);
            const wordHashDOMElement = d3.select('#'+wordHashId);
            const wordHashDOMElementBBox = wordDOMElement.node().getBBox();
            wordHashDOMElement
                .attr('x', centerXForIndex(wordIndex) - wordTextFontSize / 2)
                .attr('y', wordHashRectY + wordHashDOMElementBBox.height * 1.1);
                        
        });

        words.forEach((word, wordIndex) => {
        });
        
    };
    
    redraw();
    window.addEventListener('resize', redraw);

};

vanillaLSTMVisualizationMain();
