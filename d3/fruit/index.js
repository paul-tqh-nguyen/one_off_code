
const svg = d3.select('svg');
svg.style('background-color', 'grey');

const svg_height = parseFloat(svg.attr('height'));
const svg_width = parseFloat(svg.attr('width'));

const faceRadius = svg_height/3;
const eyeRadius = faceRadius/8;
const eyeBrowOffset = 10;
const eyeBrowWidth = 10;
const eyeTranslationMagnitude = faceRadius/3;
const eyeBrowAnimationTime = 1000;
const eyeBrowRaiseDistance = 30;

const smileyFaceGroup = svg.append('g')
      .attr('transform', `translate(${svg_width/2}, ${svg_height/2})`)
      .attr('fill', 'black');

const face = smileyFaceGroup.append('circle')
      .attr('fill', 'yellow')
      .attr('stroke-width', '5px')
      .attr('stroke', 'black')
      .attr('r', faceRadius);

const mouth = smileyFaceGroup.append('path')
      .attr('d', d3.arc()({
          innerRadius: 80,
          outerRadius: 100,
          startAngle: Math.PI/2,
          endAngle: Math.PI*3/2,
      }));

const eyesGroup = smileyFaceGroup.append('g');

const leftEye = eyesGroup.append('circle')
      .attr('transform', `translate(-${eyeTranslationMagnitude}, -${eyeTranslationMagnitude})`)
      .attr('r', eyeRadius);

const rightEye = eyesGroup.append('circle')
      .attr('transform', `translate(${eyeTranslationMagnitude}, -${eyeTranslationMagnitude})`)
      .attr('r', eyeRadius);

const leftEyeBrow = eyesGroup.append('path')
      .attr('transform', `translate(${eyeTranslationMagnitude}, -${eyeTranslationMagnitude})`)
      .attr('d', d3.arc()({
          innerRadius: eyeRadius+eyeBrowOffset,
          outerRadius: eyeRadius+eyeBrowOffset+eyeBrowWidth,
          startAngle: -Math.PI/2,
          endAngle: Math.PI/2,
      }))
      .attr('stroke-width', '5px');

const rightEyeBrow = eyesGroup.append('path')
      .attr('transform', `translate(-${eyeTranslationMagnitude}, -${eyeTranslationMagnitude})`)
      .attr('d', d3.arc()({
          innerRadius: eyeRadius+eyeBrowOffset,
          outerRadius: eyeRadius+eyeBrowOffset+eyeBrowWidth,
          startAngle: -Math.PI/2,
          endAngle: Math.PI/2,
      }))
      .attr('stroke-width', '5px');

const repeatedRaiseEyeBrows = () => {
    rightEyeBrow
        .transition()
        .duration(eyeBrowAnimationTime)
        .attr('transform', `translate(-${eyeTranslationMagnitude}, -${eyeTranslationMagnitude+eyeBrowRaiseDistance})`)
        .transition()
        .duration(eyeBrowAnimationTime)
        .attr('transform', `translate(-${eyeTranslationMagnitude}, -${eyeTranslationMagnitude})`)
        .on('end', repeatedRaiseEyeBrows);
    leftEyeBrow
        .transition()
        .duration(eyeBrowAnimationTime)
        .attr('transform', `translate(${eyeTranslationMagnitude}, -${eyeTranslationMagnitude+eyeBrowRaiseDistance})`)
        .transition()
        .duration(eyeBrowAnimationTime)
        .attr('transform', `translate(${eyeTranslationMagnitude}, -${eyeTranslationMagnitude})`)
        .on('end', repeatedRaiseEyeBrows);
};

repeatedRaiseEyeBrows();
