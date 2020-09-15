
/***************/
/* Misc. Utils */
/***************/

const isUndefined = value => value === void(0);

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

/***************************/
/* Visualization Utilities */
/***************************/

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
    'title': 'Chart of X vs Y',
    'cssFile': 'custom.css',
    'xMinValue': 0,
    'xMaxValue': 100,
    'yMinValue': 0,
    'yMaxValue': 250,
    'xAxisTitle': 'Rank',
    'yAxisTitle': 'Scores',
    'xScale': 'log',
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

:host {
  position: relative;
  width: inherit;
  height: inherit;
}

.scatter-plot-container {
  position: absolute;
  top: 0px;
  bottom: 0px;
  left: 0px;
  right: 0px;
  margin: 0px;
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
}

#tooltip.hidden{
  left: 0px;
  top: 0px;
  opacity: 0.0;
}
`});
    
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
        
        const scatterPlotGroup = svg
              .append('g')
              .classed('scatter-plot-group', true);
        const scatterPlotTitle = scatterPlotGroup.append('text');
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
        // @todo add legend

        svg
            .attr('width', scatterPlotContainer.clientWidth)
            .attr('height', scatterPlotContainer.clientHeight);
        
        const svgWidth = parseFloat(svg.attr('width'));
        const svgHeight = parseFloat(svg.attr('height'));
        
        const innerWidth = svgWidth - margin.left - margin.right;
        const innerHeight = svgHeight - margin.top - margin.bottom;

        const allPoints = [].concat(...Object.values(scatterPlotData.pointSetLookup));
        
        const xScale = scatterPlotData.xScale === 'log' ? d3.scaleLog() : d3.scaleLinear();
        xScale
            .domain([scatterPlotData.xMinValue, scatterPlotData.xMaxValue])
            .range([0, innerWidth]);
        
        const yScale = scatterPlotData.yScale === 'log' ? d3.scaleLog() : d3.scaleLinear();
        yScale
            .domain([scatterPlotData.yMaxValue, scatterPlotData.yMinValue])
            .range([0, innerHeight]);
        
        scatterPlotGroup.attr('transform', `translate(${margin.left}, ${margin.top})`);
        
        scatterPlotTitle
            .text(scatterPlotData.title)
            .attr('x', innerWidth * 0.325)
            .attr('y', -10);
        
        const xAxisTickFormat = number => d3.format('.3s')(number).replace(/G/,'B');
        xAxisGroup.call(d3.axisBottom(xScale).tickFormat(xAxisTickFormat).tickSize(-innerHeight))
            .attr('transform', `translate(0, ${innerHeight})`);
        xAxisLabel
            .attr('y', margin.bottom * 0.75)
            .attr('x', xAxisGroup.node().getBoundingClientRect().width / 2)
            .text(scatterPlotData.xAxisTitle);

        const yAxisTickFormat = number => d3.format('.3f')(number);
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

/********/
/* Main */
/********/

[
    './result_analysis/rank_0_summary.json',
    // './result_analysis/rank_1_summary.json',
    // './result_analysis/rank_2_summary.json',
    // './result_analysis/rank_3_summary.json',
    // './result_analysis/rank_4_summary.json',
    // './result_analysis/rank_5_summary.json',
    // './result_analysis/rank_6_summary.json',
    // './result_analysis/rank_7_summary.json',
    // './result_analysis/rank_8_summary.json',
    // './result_analysis/rank_9_summary.json',
].forEach((jsonFile, rank) => {
    d3.json(jsonFile)
        .then(summaryData => {

            const userScatterPlotContainer = createNewElement('div', {classes: ['user-scatter-plot-container']});
            document.querySelector('body').append(userScatterPlotContainer);
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
                'title': `Rank ${rank}User Mean MSE Loss vs User Example Count`,
                'cssFile': 'index.css',
                'xMinValue': Math.min(...userExampleCounts),
                'xMaxValue': Math.max(...userExampleCounts) + 1,
                'yMinValue': Math.min(...userMSELossValues),
                'yMaxValue': Math.max(...userMSELossValues) + 1,
                'xAxisTitle': 'Example count',
                'yAxisTitle': 'Mean MSE Loss',
                'xScale': 'log',
                'yScale': 'log',
            };
            const redrawUserScatterPlot = addScatterPlot(userScatterPlotContainer, userScatterPlotData);
            redrawUserScatterPlot();

            const animeScatterPlotContainer = createNewElement('div', {classes: ['anime-scatter-plot-container']});
            document.querySelector('body').append(animeScatterPlotContainer);
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
`,
                'pointCSSClassAccessor': pointSetName => 'anime-scatter-plot-point',
                'title': `Rank ${rank}Anime Mean MSE Loss vs Anime Example Count`,
                'cssFile': 'index.css',
                'xMinValue': Math.min(...animeExampleCounts),
                'xMaxValue': Math.max(...animeExampleCounts) + 1,
                'yMinValue': Math.min(...animeMSELossValues),
                'yMaxValue': Math.max(...animeMSELossValues) + 1,
                'xAxisTitle': 'Example count',
                'yAxisTitle': 'Mean MSE Loss',
                'xScale': 'log',
                'yScale': 'log',
            };
            const redrawAnimeScatterPlot = addScatterPlot(animeScatterPlotContainer, animeScatterPlotData);
            redrawAnimeScatterPlot();
            
            window.addEventListener('resize', () => {
                redrawUserScatterPlot();
                redrawAnimeScatterPlot();
            });
            
        }).catch(err => {
            console.error(err.message);
            return;
        });
});
