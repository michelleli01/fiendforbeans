import React from "react";

type AnswerBoxProps = {
  title: string;
  similarity: number;
  roast: string;
  titleDesc: string;
};

export default function AnswerBox({
  title,
  similarity,
  roast,
  titleDesc,
}: AnswerBoxProps): JSX.Element {
  return (
    <div>
      <h2 className="bean-title">{title}</h2>
      <p className="bean-similarity">
        Percent Confidence: {(similarity * 100).toFixed(2)}%
      </p>
      <p className="bean-similarity">Roast: {roast}</p>
      <p className="bean-desc">{titleDesc}</p>
    </div>
  );
}
