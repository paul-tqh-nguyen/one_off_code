
/***************/
/* Misc. Utils */
/***************/

const isUndefined = value => value === void(0);

const createSeparatedNumbeString = number => number.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');

const zip = rows => rows[0].map((_, i) => rows.map(row => row[i]));

// D3 Extensions
d3.selection.prototype.moveToFront = function() {
    return this.each(function() {
	if (this.parentNode !== null) {
	    this.parentNode.appendChild(this);
	}
    });
};

d3.selection.prototype.moveToBack = function() {
    return this.each(function() {
        const firstChild = this.parentNode.firstChild;
        if (firstChild) {
            this.parentNode.insertBefore(this, firstChild);
        }
    });
};


/**********************/
/* HTML Element Utils */
/**********************/

const removeAllChildNodes = (parent) => {
    while (parent.firstChild) {
        parent.removeChild(parent.firstChild);
    }
};

const createNewElement = (childTag, {classes, attributes, innerHTML}={}) => {
    const newElement = childTag === 'svg' ? document.createElementNS('http://www.w3.org/2000/svg', childTag) : document.createElement(childTag);
    if (!isUndefined(classes)) {
        classes.forEach(childClass => newElement.classList.add(childClass));
    }
    if (!isUndefined(attributes)) {
        Object.entries(attributes).forEach(([attributeName, attributeValue]) => {
            newElement.setAttribute(attributeName, attributeValue);
        });
    }
    if (!isUndefined(innerHTML)) {
        newElement.innerHTML = innerHTML;
    }
    return newElement;
};

const createTableWithElements = (rows, {classes, attributes}={}) => {
    const table = createNewElement('table', {classes, attributes});
    rows.forEach(elements => {
        const tr = document.createElement('tr');
        table.append(tr);
        elements.forEach(element => {
            const td = document.createElement('td');
            tr.append(td);
            td.append(element);
        });
    });
    return table;
};

const createLazyAccordion = (labelGeneratorDestructorTriples) => {
    const accordionContainer = createNewElement('div', {classes: ['accordion-container']});
    labelGeneratorDestructorTriples.forEach(([labelInnerHTML, contentGenerator, contentDestructor], i) => {
        // contentGenerator and contentDestructor take an HTML element
        const labelContentPairElement = createNewElement('div');
        const label = createNewElement('p', {classes: ['accordion-label'], innerHTML: labelInnerHTML});
        const contentDiv = createNewElement('div', {classes: ['accordion-content']});
        labelContentPairElement.append(label);
        labelContentPairElement.append(contentDiv);
        label.onclick = () => {
            label.classList.toggle('active');
            contentDiv.classList.toggle('active');
            if (label.classList.contains('active')) {
                contentGenerator(contentDiv);
            } else {
                contentDestructor(contentDiv);
            }
        };
        accordionContainer.append(labelContentPairElement);
    });
    return accordionContainer;
};

/***************************/
/* Visualization Utilities */
/***************************/

const d3ScaleFromString = scaleString =>
      (scaleString === 'log10') ? d3.scaleLog() :
      (scaleString === 'log2') ? d3.scaleLog().base(2) :
      (scaleString === 'squareroot') ? d3.scaleSqrt() :
      d3.scaleLinear();

const addBarChart = (container, barChartData) => {
    /* 

barChartData looks like this:
{
      'labelData': [
          {'label': 'l1', 'value': 100},
          {'label': 'l2', 'value': 200},
      ],
      'labelAccessor': datum => datum.label,
      'valueAccessor': datum => datum.value,
      'toolTipHTMLGenerator': datum => `<p>Value: ${datum.value}</p>`,
      'barCSSClassAccessor': barLabel => {
          return {
              'l1': 'l1-bar',
              'l2': 'l2-bar',
          }[barLabel];
      },
      'additionalStylesString': `
          .l2-bar {
              fill: red;
          }
      `,
      'title': 'Measurement Histogram',
      'cssFile': 'custom.css',
      'yMinValue': 0,
      'yMaxValue': 250,
      'xAxisTitle': 'Name',
      'yAxisTitle': 'Measurement',
      'yScale': 'log10',
}

This returns a re-render function, but does not actually call the re-render function initially.

*/
    
    /* Visualization Aesthetic Parameters */
    
    const margin = {
        top: 80,
        bottom: 80,
        left: 100,
        right: 30,
    };

    /* Visualization Initialization */

    removeAllChildNodes(container);
    const shadowContainer = createNewElement('div');
    container.append(shadowContainer);
    const shadow = shadowContainer.attachShadow({mode: 'open'});

    const shadowStyleElement = createNewElement('style', {innerHTML: `

@import url('https://fonts.googleapis.com/css?family=Oxygen');

:host {
  position: relative;
  width: inherit;
  height: inherit;
  font-family: 'Oxygen',  sans-serif;
}

.bar-chart-container {
  position: absolute;
  top: 0px;
  bottom: 0px;
  left: 0px;
  right: 0px;
  margin: 0px;
}

.bar-chart-group {
  transform: translate(${margin.left}px, ${margin.top}px);
}

.x-axis-group .tick line, .y-axis-group .tick line {
  opacity: 0.1;
}

.x-axis-group .tick text {
  transform: translate(0.0px, 5.0px);
}

.y-axis-group .axis-label {
  transform: rotate(-90deg);
}

.axis-label {
  fill: black;
  font-size: 1.25em;
}

#tooltip {
  position: fixed;
  transition: all 0.5s;
  text-align: center;
  font-size: 0.75em;
  background: #182A39;
  border-radius: 8px;
  pointer-events: none;
  color: #fff;
  opacity: 0.9;
  padding: 10px;
  max-width: 100vw;
  white-space: nowrap;
  overflow: hidden;
}

#tooltip.hidden{
  left: 0px;
  top: 0px;
  opacity: 0.0;
}

#tooltip p {
  margin: 0px;
  padding: 0px;
  font-size: 12px;
  font-family: inherit;
}

` + barChartData.additionalStylesString});
    
    shadow.append(shadowStyleElement);
    
    const styleInheritanceLinkElement = document.createElement('link');
    styleInheritanceLinkElement.setAttribute('rel', 'stylesheet');
    styleInheritanceLinkElement.setAttribute('href', barChartData.cssFile);
    shadow.append(styleInheritanceLinkElement);
    
    const barChartContainer = createNewElement('div', {classes: ['bar-chart-container']});
    shadow.append(barChartContainer);
    
    const svg = d3.select(barChartContainer).append('svg');
    
    const tooltipDivDomElement = createNewElement('div', {classes: ['hidden'], attributes: {'id': 'tooltip'}});
    barChartContainer.append(tooltipDivDomElement);
    const tooltipDiv = d3.select(tooltipDivDomElement);
    
    const render = () => {
        svg.selectAll('*').remove();
        
        const barChartGroup = svg
              .append('g')
              .classed('bar-chart-group', true);
        const barChartTitle = barChartGroup
              .append('text');
        const barsGroup = barChartGroup
              .append('g')
              .classed('bars-group', true);
        const xAxisGroup = barChartGroup
              .append('g')
              .classed('x-axis-group', true);
        const xAxisLabel = xAxisGroup
              .append('text')
              .classed('axis-label', true);
        const yAxisGroup = barChartGroup
              .append('g')
              .classed('y-axis-group', true);
        const yAxisLabel = yAxisGroup
              .append('text')
              .classed('axis-label', true);
        
        svg
            .attr('width', barChartContainer.clientWidth)
            .attr('height', barChartContainer.clientHeight);
        
        const svgWidth = parseFloat(svg.attr('width'));
        const svgHeight = parseFloat(svg.attr('height'));
        
        const innerWidth = svgWidth - margin.left - margin.right;
        const innerHeight = svgHeight - margin.top - margin.bottom;

        barChartTitle
            .text(barChartData.title)
            .attr('x', innerWidth / 2 - barChartTitle.node().getBBox().width / 2)
            .attr('y', -10);
        
        const xScale = d3.scaleBand()
              .domain(barChartData.labelData.map(barChartData.labelAccessor))
              .range([0, innerWidth]);

        const yScale = d3ScaleFromString(barChartData.yScale)
              .domain([barChartData.yMaxValue, barChartData.yMinValue])
              .range([0, innerHeight]);
        
        xAxisGroup.call(d3.axisBottom(xScale).tickSize(-innerHeight))
            .attr('transform', `translate(0, ${innerHeight})`);
        xAxisGroup.selectAll('.tick line').remove();
        
        yAxisGroup.call(d3.axisLeft(yScale).tickSize(-innerWidth));
        yAxisGroup.selectAll('.tick line')
            .attr('x', margin.left - 10);
        yAxisLabel
            .attr('y', -60)
            .attr('x', -innerHeight/3)
            .text(barChartData.yAxisTitle);

        xAxisLabel
            .attr('y', margin.bottom * 0.75)
            .attr('x', xAxisGroup.node().getBoundingClientRect().width / 2)
            .text(barChartData.xAxisTitle);
        
        const yAxisTickFormat = number => d3.format(',')(number);
        yAxisGroup.call(d3.axisLeft(yScale).tickFormat(yAxisTickFormat).tickSize(-innerWidth));
        
        barsGroup.selectAll('rect')
            .data(barChartData.labelData)
            .enter()
            .append('rect')
            .attr('y', datum => yScale(barChartData.valueAccessor(datum)))
            .attr('x', datum => xScale(barChartData.labelAccessor(datum)))
            .attr('width', xScale.bandwidth())
            .attr('height', datum => innerHeight-yScale(barChartData.valueAccessor(datum)))
            .on('mousemove', function(datum) {
                const [x, y] = d3.mouse(this);
                const htmlString = barChartData.toolTipHTMLGenerator(datum);
                const tooltipBoundingBox = tooltipDiv.node().getBoundingClientRect();
                const tooltipWidth = tooltipBoundingBox.width;
                const tooltipHeight = tooltipBoundingBox.height;
		tooltipDiv
		    .classed('hidden', false)
		    .style('left', x + margin.left + 'px')
		    .style('top', y + margin.top + 'px')
		    .html(htmlString);
	    })
            .on('mouseout', datum => {
		tooltipDiv
		    .classed('hidden', true);
            })
            .attr('class', datum => barChartData.barCSSClassAccessor(barChartData.labelAccessor(datum)));
        
        
    };
    
    return render;
};

/*************/
/* Color Map */
/*************/

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

/********/
/* Main */
/********/

d3.json('./aggregated_results.json')
    .then(aggregatedResults => {
        const resultsByArchitectureName = aggregatedResults.reduce((accumulator, resultData) => {
            if (!accumulator.hasOwnProperty(resultData.model_name)) {
                accumulator[resultData.model_name] = [];
            }
            accumulator[resultData.model_name].push(resultData);
            return accumulator;
        }, {});
        
        const colorMap = createRainbowColormap(Object.keys(resultsByArchitectureName).length);
        
        
        const labelGeneratorDestructorTriples = Object.entries(resultsByArchitectureName)
              .sort(([architectureNameA, modelDataA], [architectureNameB, modelDataB]) => architectureNameA < architectureNameB ? -1 : 1)
              .map(([architectureName, modelDataUnsorted], architectureIndex) => {

                  const modelData = modelDataUnsorted.sort((a,b) => a.accuracy_per_example - b.accuracy_per_example);
                  
                  let renderContent;
                  
                  const labelInnerHTML = architectureName;
                  
                  const contentGenerator = contentContainer => {
                      const testingAccuracyBarChartContainer = createNewElement('div', {classes: ['testing-accuracy-bar-chart-container', 'container-center']});
                      contentContainer.append(testingAccuracyBarChartContainer);
                      const additionalStylesString = `.testing-accuracy-bar-${architectureName} {fill: ${colorMap[architectureIndex]}; stroke-width: 1; stroke: black}`;
                      const architectureScoreData = {
                          'labelData': modelData,
                          'labelAccessor': datum => `#${modelData.indexOf(datum)}`,
                          'valueAccessor': datum => datum.accuracy_per_example,
                          'toolTipHTMLGenerator': datum => `
<p>Best Validation Epoch: ${datum.best_validation_epoch}</p>
<p>Number of Training Epochs: ${datum.number_of_epochs}</p>
<p>Testing Acccuracy Per Examplle: ${datum.accuracy_per_example}</p>
<p>Testing Loss Per Example: ${datum.loss_per_example}</p>
<p>Batch Size: ${datum.batch_size}</p>
<p>Learning Rate: ${datum.learning_rate}</p>
<p>Max Sequence Length: ${datum.max_sequence_length}</p>
<p>Gradient Clipping Max Threshold: ${datum.gradient_clipping_max_threshold}</p>
`,
                          'barCSSClassAccessor': barLabel => `testing-accuracy-bar-${architectureName}`,
                          'additionalStylesString': additionalStylesString,
                          'barAttributes': barLabel => {
                          },
                          'title': `Testing Accuracy Per Example for ${architectureName}`,
                          'cssFile': 'index.css',
                          'yMinValue': 0,
                          'yMaxValue': 1,
                          'xAxisTitle': 'Models (hover over bars for model details)',
                          'yAxisTitle': 'Testing Accuracy Per Example',
                          'yScale': 'linear',
                      };
                      const redrawBarChart = addBarChart(testingAccuracyBarChartContainer, architectureScoreData);
                      
                      renderContent = redrawBarChart;
                      
                      renderContent();
                      window.addEventListener('resize', renderContent);
                  };
                  
                  const contentDestructor = contentContainer => {
                      window.removeEventListener('resize', renderContent);
                      removeAllChildNodes(contentContainer);
                  };

                  return [labelInnerHTML, contentGenerator, contentDestructor];
              });
        
        const resultDiv = document.querySelector('#result-accordion');
        const accordion = createLazyAccordion(labelGeneratorDestructorTriples);
        accordion.classList.add('container-center');
        resultDiv.append(accordion);
    }).catch(err => {
        console.error(err.message);
        return;
    });
