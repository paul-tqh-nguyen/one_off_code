
/***************/
/* Misc. Utils */
/***************/

const isUndefined = value => value === void(0);

const createSeparatedNumbeString = number => number.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");

const zip = rows => rows[0].map((_, i) => rows.map(row => row[i]));

const numberToOrdinal = (number) => {
    const onesDigit = number % 10;
    const tensDigit = number % 100;
    if (onesDigit == 1 && tensDigit != 11) {
        return number + 'st';
    } else if (onesDigit == 2 && tensDigit != 12) {
        return number + 'nd';
    } else if (onesDigit == 3 && tensDigit != 13) {
        return number + 'rd';
    } else {
        return number + 'th';
    }
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

/*****************/
/* D3 Extensions */
/*****************/

d3.selection.prototype.moveToFront = function() {
    return this.each(function() {
	if (this.parentNode !== null) {
	    this.parentNode.appendChild(this);
	}
    });
};

d3.selection.prototype.moveToBack = function() {
    return this.each(function() {
        var firstChild = this.parentNode.firstChild;
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

const addScatterPlot = (container, scatterPlotData) => {
    /* 

scatterPlotData looks like this:
{
    'pointSetLookup': {
        'ps1': [{'x': 12, 'y': 31, 'name': 'Ingrid'}, {'x': 42, 'y': 25, 'name': 'Jure'}], 
        'ps2': [{'x': 94, 'y': 71, 'name': 'Philip'}, {'x': 76, 'y': 17, 'name': 'Nair'}], 
    },
    'xAccessor': datum => datum.x,
    'yAccessor': datum => datum.y,
    'toolTipHTMLGenerator': datum => `<p>Name: ${datum.name}</p>`,
    'pointCSSClassAccessor': pointSetName => {
        return {
            'ps1': 'ps1-point',
            'ps2': 'ps2-point',
        }[pointSetName];
    },
    'additionalStylesString': `
        .ps1-point {
            fill: red;
        }
    `,
    'title': 'Chart of X vs Y',
    'cssFile': 'custom.css',
    'xMinValue': 0,
    'xMaxValue': 100,
    'yMinValue': 0,
    'yMaxValue': 250,
    'xAxisTitle': 'Rank',
    'yAxisTitle': 'Scores',
    'xScale': 'log10',
    'yScale': 'linear',
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

.scatter-plot-container {
  position: absolute;
  top: 0px;
  bottom: 0px;
  left: 0px;
  right: 0px;
  margin: 0px;
  font-family: sans-serif;
}

.scatter-plot-group {
  transform: translate(${margin.left}px, ${margin.top}px);
}

.x-axis-group .tick line, .y-axis-group .tick line {
  opacity: 0.1;
}

.x-axis-group .tick text {
  transform: translate(0.0px, 5.0px);
}

.y-axis-group .tick text {
  transform: translate(-3.0px, 0.0px);
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

` + scatterPlotData.additionalStylesString});
    
    shadow.append(shadowStyleElement);
    
    const styleInheritanceLinkElement = document.createElement('link');
    styleInheritanceLinkElement.setAttribute('rel', 'stylesheet');
    styleInheritanceLinkElement.setAttribute('href', scatterPlotData.cssFile);
    shadow.append(styleInheritanceLinkElement);
    
    const scatterPlotContainer = createNewElement('div', {classes: ['scatter-plot-container']});
    shadow.append(scatterPlotContainer);
    
    const svg = d3.select(scatterPlotContainer).append('svg');
    
    const tooltipDivDomElement = createNewElement('div', {classes: ['hidden'], attributes: {'id': 'tooltip'}});
    scatterPlotContainer.append(tooltipDivDomElement);
    const tooltipDiv = d3.select(tooltipDivDomElement);
    
    const render = () => {
        svg.selectAll('*').remove();
        
        // @todo add legend
        const scatterPlotGroup = svg
              .append('g')
              .classed('scatter-plot-group', true);
        const scatterPlotTitle = scatterPlotGroup
              .append('text');
        const xAxisGroup = scatterPlotGroup
              .append('g')
              .attr('class', 'x-axis-group', true);
        const xAxisLabel = xAxisGroup
              .append('text')
              .classed('axis-label', true);
        const yAxisGroup = scatterPlotGroup
              .append('g')
              .classed('y-axis-group', true);
        const yAxisLabel = yAxisGroup
              .append('text')
              .classed('axis-label', true);

        svg
            .attr('width', scatterPlotContainer.clientWidth)
            .attr('height', scatterPlotContainer.clientHeight);
        
        const svgWidth = parseFloat(svg.attr('width'));
        const svgHeight = parseFloat(svg.attr('height'));
        
        const innerWidth = svgWidth - margin.left - margin.right;
        const innerHeight = svgHeight - margin.top - margin.bottom;

        const allPoints = [].concat(...Object.values(scatterPlotData.pointSetLookup));
        
        const xScale = d3ScaleFromString(scatterPlotData.xScale);
        xScale
            .domain([scatterPlotData.xMinValue, scatterPlotData.xMaxValue])
            .range([0, innerWidth]);
        
        const yScale = d3ScaleFromString(scatterPlotData.yScale);
        yScale
            .domain([scatterPlotData.yMaxValue, scatterPlotData.yMinValue])
            .range([0, innerHeight]);
        
        scatterPlotTitle
            .text(scatterPlotData.title)
            .attr('x', innerWidth / 2 - scatterPlotTitle.node().getBBox().width / 2)
            .attr('y', -10);
        
        const xAxisTickFormat = number => d3.format('.0s')(number).replace(/G/,'B');
        xAxisGroup.call(d3.axisBottom(xScale).tickFormat(xAxisTickFormat).tickSize(-innerHeight))
            .attr('transform', `translate(0, ${innerHeight})`);
        xAxisLabel
            .attr('y', margin.bottom * 0.75)
            .attr('x', xAxisGroup.node().getBoundingClientRect().width / 2)
            .text(scatterPlotData.xAxisTitle);

        const yAxisTickFormat = number => d3.format('.2f')(number);
        yAxisGroup.call(d3.axisLeft(yScale).tickFormat(yAxisTickFormat).tickSize(-innerWidth));
        yAxisLabel
            .attr('y', -60)
            .attr('x', -innerHeight/3)
            .text(scatterPlotData.yAxisTitle);
        
        const xAccessor = scatterPlotData.xAccessor;
        const yAccessor = scatterPlotData.yAccessor;
        
        Object.entries(scatterPlotData.pointSetLookup).forEach(([pointSetName, points]) => {
            const pointCSSClass = scatterPlotData.pointCSSClassAccessor(pointSetName);
            const pointSetGroup = scatterPlotGroup
                  .append('g')
                  .classed(`point-set-group-${pointSetName}`, true);
            pointSetGroup.selectAll('circle')
                .data(points)
                .enter()
                .append('circle')
                .on('mouseover', function(datum) {
                    const boundingBox = d3.select(this).node().getBoundingClientRect();
                    const x = boundingBox.left;
                    const y = boundingBox.top;
                    const htmlString = scatterPlotData.toolTipHTMLGenerator(datum);
		    tooltipDiv
		        .classed('hidden', false)
		        .html(htmlString)
		        .style('left', x + 10 + 'px')
		        .style('top', y + 10 + 'px');
	        })
                .on('mouseout', datum => {
		    tooltipDiv
		        .classed('hidden', true);
                })
                .classed(pointCSSClass, true)
                .attr('cx', datum => xScale(xAccessor(datum)))
                .attr('cy', datum => yScale(yAccessor(datum)));
        });
    };
    
    return render;
};

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
        
        // @todo add legend
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
            .on('mouseover', function(datum) {
                const boundingBox = d3.select(this).node().getBoundingClientRect();
                const x = boundingBox.left;
                const y = boundingBox.top;
                const htmlString = barChartData.toolTipHTMLGenerator(datum);
                const tooltipBoundingBox = tooltipDiv.node().getBoundingClientRect();
                const tooltipWidth = tooltipBoundingBox.width;
                const tooltipHeight = tooltipBoundingBox.height;
		tooltipDiv
		    .classed('hidden', false)
		    .style('left', x + boundingBox.width / 2 - tooltipWidth / 2 + 'px')
		    .style('top', y - tooltipHeight - 10 + 'px')
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

/********/
/* Main */
/********/

const generateLabelGeneratorDestructorTriples = (architectureName, rank, summaryData, animeLookupById, additionalStylesString) => {
    let renderContent;
    const labelInnerHTML = `${architectureName} ${numberToOrdinal(rank)} Place`;
    const contentGenerator = (contentContainer) => {

        const hyperparameterTableContainer = createNewElement('div', {classes: ['hyperparameter-table-container']});
        contentContainer.append(hyperparameterTableContainer);
        const renderHyperparameterTable = () => {
            removeAllChildNodes(hyperparameterTableContainer);
            hyperparameterTableContainer.append(createNewElement('p', {innerHTML: 'Results', attributes: {style: 'margin: 2em 0px 10px 0px'}}));
            hyperparameterTableContainer.append(
                createTableWithElements([
                    [createNewElement('p', {innerHTML: `Testing MSE Loss:`}), createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: `${summaryData.testing_mse_loss}`})],
                    [createNewElement('p', {innerHTML: `Best Validation Loss:`}), createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: ` ${summaryData.best_validation_loss}`})],
                ], {classes: ['hyperparameter-table'], attributes: {style: 'margin-bottom: 1em;'}})
            );
            hyperparameterTableContainer.append(createNewElement('p', {innerHTML: 'Hyperparameters', attributes: {style: 'margin-bottom: 10px'}}));
            const rows = [
                [createNewElement('p', {innerHTML: `Learning Rate:`}), createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: ` ${summaryData.learning_rate}`})],
                [createNewElement('p', {innerHTML: `Number of Training Epochs:`}), createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: ` ${summaryData.number_of_epochs}`})],
                [createNewElement('p', {innerHTML: `Batch Size:`}), createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: ` ${summaryData.batch_size}`})],
                [createNewElement('p', {innerHTML: `Embedding Size:`}), createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: ` ${summaryData.embedding_size}`})],
            ];
            if (summaryData.hasOwnProperty('dense_layer_count')) {
                rows.push([createNewElement('p', {innerHTML: `Number of Dense Layers:`}), createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: ` ${summaryData.dense_layer_count}`})]);
            }
            rows.push([createNewElement('p', {innerHTML: `Regularization Factor:`}), createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: ` ${summaryData.regularization_factor}`})]);
            rows.push([createNewElement('p', {innerHTML: `Dropout Probability:`}), createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: ` ${summaryData.dropout_probability}`})]);
            hyperparameterTableContainer.append(
                createTableWithElements(rows, {classes: ['hyperparameter-table']})
            );
        };
        
        const roundedScoreToUserCount = Object.entries(summaryData.user_data).reduce((accumulator, [userId, datum]) => {
            const roundedMSELoss = Math.round(datum.mean_mse_loss);
            if (!(accumulator.hasOwnProperty(roundedMSELoss))) {
                accumulator[roundedMSELoss] = 0;
            }
            accumulator[roundedMSELoss] += 1;
            return accumulator;
        }, {});
        for(let i=0; i<Object.keys(roundedScoreToUserCount).reduce((a, b) => Math.max(parseInt(a), b), 0); i++) {
            if (!(roundedScoreToUserCount.hasOwnProperty(i.toString()))) {
                roundedScoreToUserCount[i.toString()] = 0;
            }
        }
        const roundedScoreHistogramContainer = createNewElement('div', {classes: ['rounded-score-histogram-container']});
        contentContainer.append(roundedScoreHistogramContainer);
        const roundedScoreHistogramData = {
            'labelData': Object.entries(roundedScoreToUserCount).map(([roundedMSELoss, userCount]) => {
                return {'userCount': userCount, 'roundedMSELoss': roundedMSELoss};
            }),
            'labelAccessor': datum => datum.roundedMSELoss,
            'valueAccessor': datum => datum.userCount,
            'toolTipHTMLGenerator': datum => `<p>User Count: ${datum.userCount}</p><p>Rounded MSE Loss: ${datum.roundedMSELoss}</p>`,
            'barCSSClassAccessor': barLabel => 'histogram-bar',
            'additionalStylesString': additionalStylesString,
            'title': 'User Count vs MSE Loss Histogram',
            'cssFile': 'index.css',
            'yMinValue': Math.min(...Object.values(roundedScoreToUserCount)) / 2,
            'yMaxValue': Math.max(...Object.values(roundedScoreToUserCount)) + 1000,
            'xAxisTitle': 'Rounded MSE Loss',
            'yAxisTitle': 'User Count (Squareroot Scale)',
            'yScale': 'squareroot',
        };
        const redrawBarChart = addBarChart(roundedScoreHistogramContainer, roundedScoreHistogramData);
        

        const userScatterPlotContainer = createNewElement('div', {classes: ['user-scatter-plot-container']});
        contentContainer.append(userScatterPlotContainer);
        const userExampleCounts = Object.values(summaryData.user_data).map(datum => datum.example_count);
        const userMSELossValues = Object.values(summaryData.user_data).map(datum => datum.mean_mse_loss);
        const userScatterPlotData = {
            'pointSetLookup': {
                'users': Object.entries(summaryData.user_data).map(([userId, userData]) => Object.assign(userData, {'id': userId})),
            },
            'xAccessor': datum => datum.example_count,
            'yAccessor': datum => datum.mean_mse_loss,
            'toolTipHTMLGenerator': datum => `
<p>User Id: ${datum.id}</p>
<p>Total MSE Loss: ${datum.total_mse_loss}</p>
<p>Mean MSE Loss: ${datum.mean_mse_loss}</p>
<p>Example Count: ${datum.example_count}</p>
`,
            'pointCSSClassAccessor': pointSetName => 'user-scatter-plot-point',
            'additionalStylesString': additionalStylesString,
            'title': `User Mean MSE Loss vs User Example Count`,
            'cssFile': 'index.css',
            'xMinValue': Math.min(...userExampleCounts) / 2,
            'xMaxValue': Math.max(...userExampleCounts) + 100,
            'yMinValue': Math.min(...userMSELossValues) / 2,
            'yMaxValue': Math.max(...userMSELossValues) + 10,
            'xAxisTitle': 'Example count',
            'yAxisTitle': 'Mean MSE Loss',
            'xScale': 'log10',
            'yScale': 'log10',
        };
        const redrawUserScatterPlot = addScatterPlot(userScatterPlotContainer, userScatterPlotData);

        const animeScatterPlotContainer = createNewElement('div', {classes: ['anime-scatter-plot-container']});
        contentContainer.append(animeScatterPlotContainer);
        const animeExampleCounts = Object.values(summaryData.anime_data).map(datum => datum.example_count);
        const animeMSELossValues = Object.values(summaryData.anime_data).map(datum => datum.mean_mse_loss);
        const animeScatterPlotData = {
            'pointSetLookup': {
                'animes': Object.entries(summaryData.anime_data).map(([animeId, animeData]) => Object.assign(animeData, {'id': animeId})),
            },
            'xAccessor': datum => datum.example_count,
            'yAccessor': datum => datum.mean_mse_loss,
            'toolTipHTMLGenerator': datum => `
<p>Anime Id: ${datum.id}</p>
<p>Total MSE Loss: ${datum.total_mse_loss}</p>
<p>Mean MSE Loss: ${datum.mean_mse_loss}</p>
<p>Example Count: ${datum.example_count}</p>
<p></p>
<p>Anime Name: ${animeLookupById[datum.id].name}</p>
<p>Genre: ${animeLookupById[datum.id].genre}</p>
<p>Anime Type: ${animeLookupById[datum.id].type}</p>
<p>Episode Count: ${animeLookupById[datum.id].episodes}</p>
`,
            'pointCSSClassAccessor': pointSetName => 'anime-scatter-plot-point',
            'additionalStylesString': additionalStylesString,
            'title': `Anime Mean MSE Loss vs Anime Example Count`,
            'cssFile': 'index.css',
            'xMinValue': Math.min(...animeExampleCounts) / 2,
            'xMaxValue': Math.max(...animeExampleCounts) + 100,
            'yMinValue': Math.min(...animeMSELossValues) / 2,
            'yMaxValue': Math.max(...animeMSELossValues) + 10,
            'xAxisTitle': 'Example count',
            'yAxisTitle': 'Mean MSE Loss',
            'xScale': 'log10',
            'yScale': 'log10',
        };
        const redrawAnimeScatterPlot = addScatterPlot(animeScatterPlotContainer, animeScatterPlotData);
        
        renderContent = () => {
            renderHyperparameterTable();
            redrawBarChart();
            redrawUserScatterPlot();
            redrawAnimeScatterPlot();
        };
        renderContent();
        window.addEventListener('resize', renderContent);
    };
    const contentDestructor = (contentContainer) => {
        window.removeEventListener('resize', renderContent);
        removeAllChildNodes(contentContainer);
    };

    return [labelInnerHTML, contentGenerator, contentDestructor];
};

const colorMap = createRainbowColormap(6).reverse();

d3.csv("./anime.csv").then(
    animeCSVData =>
        animeCSVData.reduce((accumulator, row) => {
            accumulator[row.anime_id] = row;
            delete row.anime_id;
            return accumulator;
        }, {})
).then((animeLookupById) => Promise.all(
    [
        './result_analysis/LinearColaborativeFilteringModel_rank_0_summary.json',
        './result_analysis/LinearColaborativeFilteringModel_rank_1_summary.json',
        './result_analysis/LinearColaborativeFilteringModel_rank_2_summary.json',
        './result_analysis/LinearColaborativeFilteringModel_rank_3_summary.json',
        './result_analysis/LinearColaborativeFilteringModel_rank_4_summary.json',
        './result_analysis/LinearColaborativeFilteringModel_rank_5_summary.json',
        './result_analysis/LinearColaborativeFilteringModel_rank_6_summary.json',
        './result_analysis/LinearColaborativeFilteringModel_rank_7_summary.json',
        './result_analysis/LinearColaborativeFilteringModel_rank_8_summary.json',
        './result_analysis/LinearColaborativeFilteringModel_rank_9_summary.json',
    ].map((jsonFile, jsonFileIndex) => d3.json(jsonFile)
          .then(summaryData => generateLabelGeneratorDestructorTriples('Linear Model', jsonFileIndex+1, summaryData, animeLookupById, `
.user-scatter-plot-point {
    fill: ${colorMap[0]};
}

.anime-scatter-plot-point {
    fill: ${colorMap[2]};
}

.histogram-bar {
    fill: ${colorMap[4]};
}
`)))
).then((labelGeneratorDestructorTriples) => {
    const resultDiv = document.querySelector('#linear-result');
    const accordion = createLazyAccordion(labelGeneratorDestructorTriples);
    resultDiv.append(accordion);
}).catch(err => {
    console.error(err.message);
    return;
}));

d3.csv("./anime.csv").then(
    animeCSVData =>
        animeCSVData.reduce((accumulator, row) => {
            accumulator[row.anime_id] = row;
            delete row.anime_id;
            return accumulator;
        }, {})
).then((animeLookupById) => Promise.all(
    [
        './result_analysis/DeepConcatenationColaborativeFilteringModel_rank_0_summary.json',
        './result_analysis/DeepConcatenationColaborativeFilteringModel_rank_1_summary.json',
        './result_analysis/DeepConcatenationColaborativeFilteringModel_rank_2_summary.json',
        './result_analysis/DeepConcatenationColaborativeFilteringModel_rank_3_summary.json',
        './result_analysis/DeepConcatenationColaborativeFilteringModel_rank_4_summary.json',
        './result_analysis/DeepConcatenationColaborativeFilteringModel_rank_5_summary.json',
        './result_analysis/DeepConcatenationColaborativeFilteringModel_rank_6_summary.json',
        './result_analysis/DeepConcatenationColaborativeFilteringModel_rank_7_summary.json',
        './result_analysis/DeepConcatenationColaborativeFilteringModel_rank_8_summary.json',
        './result_analysis/DeepConcatenationColaborativeFilteringModel_rank_9_summary.json',
    ].map((jsonFile, jsonFileIndex) => d3.json(jsonFile)
          .then(summaryData => generateLabelGeneratorDestructorTriples('Deep Model', jsonFileIndex+1, summaryData, animeLookupById, `
.user-scatter-plot-point {
    fill: ${colorMap[1]};
}

.anime-scatter-plot-point {
    fill: ${colorMap[3]};
}

.histogram-bar {
    fill: ${colorMap[5]};
}
`)))
).then((labelGeneratorDestructorTriples) => {
    const resultDiv = document.querySelector('#deep-result');
    const accordion = createLazyAccordion(labelGeneratorDestructorTriples);
    resultDiv.append(accordion);
}).catch(err => {
    console.error(err.message);
    return;
}));
