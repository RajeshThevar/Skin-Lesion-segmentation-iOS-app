//
//  TouchDrawView.h
//  OpenCVTest
//
//  Created by RAJESH on 3/10/16.
//  Copyright © 2016 RAJESH. All rights reserved.

//

#import <UIKit/UIKit.h>

typedef enum TouchState{
    TouchStateNone,
    TouchStateRect,
    TouchStatePlus,
    TouchStateMinus
}TouchState;


@interface TouchDrawView : UIView{
    CGPoint pts[5]; // we now need to keep track of the four points of a Bezier segment and the first control point of the next segment
    uint ctr;
}
@property (nonatomic, assign) TouchState currentState;
- (void) touchStarted:(CGPoint) p;
- (void) touchMoved:(CGPoint) p;
- (void) touchEnded:(CGPoint) p;
- (void) drawRectangle:(CGRect) rect;
- (void) clear;
- (UIImage *) maskImageWithPainting;
@end
