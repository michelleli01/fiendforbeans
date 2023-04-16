import React from "react";
import Bean from "../types";
import AnswerBox from "./AnswerBox";

type RecommendationsProps = { recommend: Bean[] | undefined };

export default function Recommendations({
  recommend,
}: RecommendationsProps): JSX.Element {
  return (
    <div>
      {recommend === undefined ? (
        <></>
      ) : recommend.length === 0 ? (
        <div>
          <p>No coffee beans match your query. Please try again</p>
        </div>
      ) : (
        <div>
          {recommend.map((bean) => (
            <AnswerBox
              key={bean.bean_info.name}
              title={bean.bean_info.name}
              roast={bean.bean_info.roast}
              similarity={bean.score}
              titleDesc={bean.bean_info.review}
            />
          ))}
        </div>
      )}
    </div>
  );
}
