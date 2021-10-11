const isUndefined = value => value === void(0);

const zip = rows => rows[0].map((_, i) => rows.map(row => row[i]));

const lerp = (start, end, interpolationAmount) => start + interpolationAmount * (end - start);

const createRainbowColormap = (shadeCount) => {

    const rainbowMap = [
        {'amount': 0,      'rgb':[150, 0, 90]},
        {'amount': 0.125,  'rgb': [0, 0, 200]},
        {'amount': 0.25,   'rgb': [0, 25, 255]},
        {'amount': 0.375,  'rgb': [0, 152, 255]},
        {'amount': 0.5,    'rgb': [44, 255, 150]},
        {'amount': 0.625,  'rgb': [151, 255, 0]},
        {'amount': 0.75,   'rgb': [255, 234, 0]},
        {'amount': 0.875,  'rgb': [255, 111, 0]},
        {'amount': 1,      'rgb': [255, 0, 0]}
    ];

    const colors = [];
    for (let i = 0; i < shadeCount; i++) {
        const rgbStartIndex = Math.floor((rainbowMap.length-1) * i/(shadeCount-1));
        const rgbEndIndex = Math.ceil((rainbowMap.length-1) * i/(shadeCount-1));
        const rgbStart = rainbowMap[rgbStartIndex].rgb;
        const rgbEnd = rainbowMap[rgbEndIndex].rgb;
        const interpolationRange = rainbowMap[rgbEndIndex].amount - rainbowMap[rgbStartIndex].amount;
        const interpolationAmount = interpolationRange === 0 ? 0 : (i/(shadeCount-1) - rainbowMap[rgbStartIndex].amount) / interpolationRange;
        const rgbInterpolated = zip([rgbStart, rgbEnd]).map(([rgbStartChannel, rgbEndChannel]) => Math.round(lerp(rgbStartChannel, rgbEndChannel, interpolationAmount)));
        const hex = '#' + rgbInterpolated.map(channel => channel.toString(16).padStart(2, '0')).join('');
        colors.push(hex);
    }
    return colors;
};

// D3 Extensions
d3.selection.prototype.moveToFront = function() {
    return this.each(function() {
        if (this.parentNode !== null) {
            this.parentNode.appendChild(this);
        }
    });
};

const visualizationMain = () => {
    const dataLocation = './bytecode.json';

    d3.json(dataLocation)
        .then(data => {
            document.getElementById('header-sub-title').innerHTML =
                `Function <code>${data.func_name}</code> from <code>${data.func_file_location}</code>`;

            const sourceCodeLineNumberToBasicBlockId = {};
            const basicBlockIdToBasicBlockDict = {};
            data.nodes.forEach((basicBlockDict, basicBlockIndex) => {
                basicBlockDict.sequentialIndex = basicBlockIndex;
                basicBlockIdToBasicBlockDict[basicBlockDict.id] = basicBlockDict;
                basicBlockDict.source_code_line_numbers.forEach(sourceCodeLineNumber => {
                    sourceCodeLineNumberToBasicBlockId[sourceCodeLineNumber] = basicBlockDict.id;
                });
                if (basicBlockDict.source_code_line_numbers.length == 0) {
                    console.error("Cannot yet handle basic blocks without corresponding source code lines.");
                }
            });

            const codeDisplayElement = document.getElementById('code-table');
            let sourceCodeLineNumber = data.source_code_line_number;
            let previousBasicBlockId = -1;
            let previousBytecodeTdElement;
            const lineNumberWidth = Math.ceil(Math.log(data.source_code_lines.length + sourceCodeLineNumber));
            const numLeadingSpaces = data.source_code_lines[0].search(/\S/);
            const colorMap = createRainbowColormap(data.nodes.length+1);
            data.source_code_lines.forEach(sourceLine => {
                const rowElement = document.createElement('tr');
    
                const sourceTdElement = document.createElement('td');
                const sourceDivElement = document.createElement('div');
                const sourcePreElement = document.createElement('pre');
                sourcePreElement.innerHTML =
                    String(sourceCodeLineNumber).padStart(lineNumberWidth, '0')
                    + ' '
                    + sourceLine.slice(numLeadingSpaces);
                sourceDivElement.append(sourcePreElement);
                sourceTdElement.append(sourceDivElement);
                rowElement.append(sourceTdElement);
    
                let colorIndex = data.nodes.length+1;
                const bytecodeTdElement = document.createElement('td');
                const bytecodeDivElement = document.createElement('div');
                bytecodeTdElement.append(bytecodeDivElement);
                if (sourceCodeLineNumber in sourceCodeLineNumberToBasicBlockId) {
                    const basicBlockId = sourceCodeLineNumberToBasicBlockId[sourceCodeLineNumber];
                    colorIndex = basicBlockIdToBasicBlockDict[basicBlockId].sequentialIndex;
                    if (previousBasicBlockId == basicBlockId) {
                        const newRowSpanValue = parseInt(previousBytecodeTdElement.getAttribute('rowspan')) + 1;
                        previousBytecodeTdElement.setAttribute('rowspan', newRowSpanValue);
                    } else {
                        previousBasicBlockId = basicBlockId;
                        basicBlockIdToBasicBlockDict[basicBlockId].pretty_strings.forEach(prettyString => {
                            const bytecodePreElement = document.createElement('pre');
                            bytecodePreElement.innerHTML = prettyString;
                            bytecodeDivElement.append(bytecodePreElement);
                        });
                        bytecodeTdElement.setAttribute('rowspan', 1);
                        rowElement.append(bytecodeTdElement);
                        previousBytecodeTdElement = bytecodeTdElement;
                    }
                } else if (!isUndefined(previousBytecodeTdElement)) {
                    const newRowSpanValue = parseInt(previousBytecodeTdElement.getAttribute('rowspan')) + 1;
                    previousBytecodeTdElement.setAttribute('rowspan', newRowSpanValue);
                }
                codeDisplayElement.append(rowElement);
    
                const rowColor = colorMap[colorIndex];
                sourceTdElement.style.borderColor = rowColor;
                // FIXME sometimes this is useless when we don't use the td element
                bytecodeTdElement.style.borderColor = rowColor;
    
                sourceCodeLineNumber++;
            });

            const plotContainer = document.getElementById('cfg-container');
            const svg = d3.select('#cfg-svg');

            const render = () => {

                const basicBlockTextBoundingBoxPadding = 15;
    
                svg
	            .attr('width', `0px`)
	            .attr('height', `0px`)
	            .attr('width', `${plotContainer.clientWidth}px`)
	            .attr('height', `${plotContainer.clientHeight}px`);
    
                const basicBlockGroup = svg.append('g');

                const basicBlockDataJoin = basicBlockGroup
                      .selectAll('text')
                      .data(data.nodes);

                const basicBlockTextEnterSelection = basicBlockDataJoin
                      .enter()
                      .append('text')
                      .attr('id', datum => `basic-block-${datum.id}`)
                      .attr('class', 'basic-block')
                      .attr('x', datum => {
                          return 10*(2+datum.id);
                      })
                      .attr('y', datum => {
                          return 10*(2+datum.id);
                      })
                      .html(datum => {
                          const textElementId = `basic-block-${datum.id}`;
                          const textElementBBox = svg.select(`#${textElementId}`).node().getBBox();
                          return datum.pretty_strings.map(
                              string => `<tspan x=${textElementBBox.x} dx=0 dy=22>${string}</tspan>`
                          ).join('\n');
                      });

                const  basicBlockBoundingBoxEnterSelection = basicBlockDataJoin
                      .enter()
                      .append('rect')
                      .attr('id', datum => `basic-block-bounding-box-{datum.id}`)
                      .attr('stroke', datum => colorMap[datum.sequentialIndex])
                      .attr('class', 'basic-block-bounding-box');

                basicBlockBoundingBoxEnterSelection
                    .attr('x', datum => {
                        const textElementId = `basic-block-${datum.id}`;
                        const textElementBBox = svg.select(`#${textElementId}`).node().getBBox();
                        const x = textElementBBox.x - basicBlockTextBoundingBoxPadding;
                        return x;
                    })
                    .attr('y', datum => {
                        const textElementId = `basic-block-${datum.id}`;
                        const textElementBBox = svg.select(`#${textElementId}`).node().getBBox();
                        const y = textElementBBox.y - basicBlockTextBoundingBoxPadding;
                        return y;
                    })
                    .attr('width', datum => {
                        const textElementId = `basic-block-${datum.id}`;
                        const textElementBBox = svg.select(`#${textElementId}`).node().getBBox();
                        return textElementBBox.width + 2 * basicBlockTextBoundingBoxPadding;
                    })
                    .attr('height', datum => {
                        const textElementId = `basic-block-${datum.id}`;;
                        const textElementBBox = svg.select(`#${textElementId}`).node().getBBox();
                        return textElementBBox.height + 2 * basicBlockTextBoundingBoxPadding;
                    });

                basicBlockTextEnterSelection.moveToFront();
            };
            render();
            window.addEventListener('resize', render);


        }).catch(err => {
            console.error(err.message);
            return;
        });

};

visualizationMain();
